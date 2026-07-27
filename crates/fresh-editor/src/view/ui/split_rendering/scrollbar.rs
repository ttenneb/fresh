//! Scrollbar computation and rendering (vertical, horizontal, composite).
//!
//! These helpers take the editor `State`, viewport, and a few typed
//! parameters. They have no dependency on any shared render-time "mega
//! struct".

use crate::primitives::visual_layout::visual_width_with_tab_size;
use crate::state::EditorState;
use crate::view::scrollbar_marker::{self, MarkerBasis, MarkerCell};
use crate::view::theme::Theme;
use crate::view::viewport::Viewport;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Paragraph;
use ratatui::widgets::Widget;

/// Above either bound, the exact wrapped-row scrollbar is skipped in favour of
/// the cheap logical-line approximation. The exact counts require word-wrapping
/// every line (`ensure_built`, O(all-lines)) and the index is rebuilt whenever
/// the buffer version changes — i.e. on every edit — so on a large wrapped
/// buffer each keystroke re-walks the whole buffer and stalls the UI
/// (fresh#2610). Both bounds are well under the 10 MB large-file threshold
/// because the per-line word-wrap cost is much higher than a raw byte scan.
/// Line count is the loop-iteration driver; the byte bound catches buffers with
/// few but very long lines.
const MAX_WRAP_SCROLLBAR_LINES: usize = 5_000;
const MAX_WRAP_SCROLLBAR_BYTES: usize = 2 * 1024 * 1024;

/// Compute scrollbar line counts: `(total_lines, top_line)`, plus the
/// coordinate basis those counts are expressed in.
///
/// For large files the counts are reported as `(0, 0)` — the caller uses a
/// constant-size thumb in that case. When line wrapping is enabled, counts are
/// in visual rows instead of logical lines — except on a large wrapped buffer,
/// where the exact visual-row count is too expensive to recompute per edit and
/// we fall back to the logical-line approximation (see the constants above).
///
/// The returned [`MarkerBasis`] is what plugin scrollbar markers are projected
/// through. Deriving it here, from the same branch that picks the thumb's
/// counts, is what keeps markers and thumb in the same coordinate space by
/// construction rather than by convention.
pub(super) fn scrollbar_line_counts(
    state: &mut EditorState,
    viewport: &Viewport,
    large_file_threshold_bytes: u64,
    buffer_len: usize,
) -> (usize, usize, MarkerBasis) {
    if buffer_len > large_file_threshold_bytes as usize {
        // No line coordinate exists here — on a file this size `line_count()`
        // is typically `None` until the incremental scan runs. Bytes are exact
        // and O(1), so markers stay correct on the very first frame.
        return (
            0,
            0,
            MarkerBasis::Bytes {
                total: buffer_len as u64,
            },
        );
    }

    let total_lines = if buffer_len > 0 {
        state.buffer.get_line_number(buffer_len.saturating_sub(1)) + 1
    } else {
        1
    };

    if viewport.line_wrap_enabled
        && total_lines <= MAX_WRAP_SCROLLBAR_LINES
        && buffer_len <= MAX_WRAP_SCROLLBAR_BYTES
    {
        let (total_rows, top_row) = scrollbar_visual_row_counts(state, viewport, buffer_len);
        return (
            total_rows,
            top_row,
            MarkerBasis::VisualRows {
                total: total_rows as u64,
            },
        );
    }

    let top_line = if viewport.top_byte < buffer_len {
        state.buffer.get_line_number(viewport.top_byte)
    } else {
        0
    };

    (
        total_lines,
        top_line,
        MarkerBasis::LogicalLines {
            total: total_lines as u64,
        },
    )
}

/// Project this buffer's plugin scrollbar markers onto a `track_height`-tall
/// column, reusing the cached projection when nothing relevant changed.
///
/// Cost per frame is one key comparison in the steady state; a rebuild is
/// O(M) in the marker count with an O(1)–O(log n) coordinate lookup each, and
/// carries no term proportional to file size. The `basis` argument comes from
/// [`scrollbar_line_counts`], so the projection always uses the same
/// coordinate space as the thumb it is drawn beside.
pub(super) fn project_scrollbar_markers(
    state: &mut EditorState,
    basis: MarkerBasis,
    track_height: usize,
) -> Vec<Option<MarkerCell>> {
    if state.scrollbar_markers.is_empty() || track_height == 0 {
        return Vec::new();
    }

    let content_version = state.buffer.version();

    // Split the borrows: the projection reads the manager and the coordinate
    // source while writing the bucket cache.
    let mut buckets = std::mem::take(&mut state.scrollbar_marker_buckets);
    let cells = match basis {
        MarkerBasis::Bytes { .. } => scrollbar_marker::project(
            &state.scrollbar_markers,
            &mut buckets,
            basis,
            track_height,
            content_version,
            |byte| byte as u64,
        )
        .to_vec(),
        MarkerBasis::LogicalLines { .. } => {
            // `get_line_number` needs `&mut Buffer`; markers are read from a
            // snapshot first so the two borrows don't overlap.
            let resolved = state.scrollbar_markers.resolved();
            let mut lines = std::collections::HashMap::with_capacity(resolved.len() * 2);
            for m in &resolved {
                lines
                    .entry(m.start)
                    .or_insert_with(|| state.buffer.get_line_number(m.start) as u64);
                if let Some(e) = m.end {
                    lines
                        .entry(e)
                        .or_insert_with(|| state.buffer.get_line_number(e) as u64);
                }
            }
            scrollbar_marker::project(
                &state.scrollbar_markers,
                &mut buckets,
                basis,
                track_height,
                content_version,
                |byte| lines.get(&byte).copied().unwrap_or(0),
            )
            .to_vec()
        }
        MarkerBasis::VisualRows { .. } => {
            // The visual-row index was just built by
            // `scrollbar_visual_row_counts` for this same frame and geometry;
            // this path only reads it, never triggers the O(all-lines) build.
            let index = &state.visual_row_index;
            scrollbar_marker::project(
                &state.scrollbar_markers,
                &mut buckets,
                basis,
                track_height,
                content_version,
                |byte| {
                    let (line_idx, _) = index.line_for_byte(byte);
                    index.line_first_row(line_idx) as u64
                },
            )
            .to_vec()
        }
    };
    state.scrollbar_marker_buckets = buckets;
    cells
}

/// Calculate scrollbar position based on visual rows (for line-wrapped content).
/// Returns `(total_visual_rows, top_visual_row)`.
///
/// Both numbers come from the per-state [`VisualRowIndex`] in O(log N_lines).
/// The index is built lazily and reused across frames whenever its key
/// (pipeline-input version + geometry) is unchanged — so a steady-state
/// scroll where only `top_byte` moves never re-walks the buffer.
///
/// [`VisualRowIndex`]: crate::view::visual_row_index::VisualRowIndex
pub(super) fn scrollbar_visual_row_counts(
    state: &mut EditorState,
    viewport: &Viewport,
    buffer_len: usize,
) -> (usize, usize) {
    use crate::primitives::line_wrapping::WrapConfig;
    use crate::view::line_wrap_cache::{pipeline_inputs_version, CacheViewMode};
    use crate::view::visual_row_index::{ensure_built, VisualRowIndexKey};

    if buffer_len == 0 {
        return (1, 0);
    }

    // Terminal-grid wrap (fresh#2649): count exact-column rows at the grid
    // width — same row model as the renderer and the viewport scroll math.
    let (effective_width, gutter_width, hanging_indent) = if viewport.grid_wrap {
        (viewport.grid_cols(), 0usize, false)
    } else {
        let gutter_width = viewport.gutter_width(&state.buffer);
        let wrap_config = WrapConfig::new(
            viewport.width as usize,
            gutter_width,
            true,
            viewport.wrap_indent,
        );
        let effective_width = wrap_config
            .first_line_width
            .saturating_add(gutter_width)
            .max(2);
        (effective_width, gutter_width, wrap_config.hanging_indent)
    };
    let pipeline_inputs_ver = pipeline_inputs_version(
        state.buffer.version(),
        state.soft_breaks.version(),
        state.conceals.version(),
        state.virtual_texts.version(),
    );

    let key = VisualRowIndexKey {
        pipeline_inputs_version: pipeline_inputs_ver,
        view_mode: CacheViewMode::Source,
        effective_width: effective_width as u32,
        gutter_width: gutter_width as u16,
        wrap_column: None,
        hanging_indent,
        line_wrap_enabled: viewport.line_wrap_enabled,
        grid_wrap: viewport.grid_wrap,
    };
    ensure_built(state, &key);

    let total_visual_rows = state.visual_row_index.total_rows() as usize;
    let total_visual_rows = total_visual_rows.max(1);

    // Top visual row: first row of the line containing `top_byte`,
    // plus the wrap-segment offset within that line.
    let (line_idx, _) = state.visual_row_index.line_for_byte(viewport.top_byte);
    let top_first_row = state.visual_row_index.line_first_row(line_idx) as usize;
    let top_visual_row =
        (top_first_row + viewport.top_view_line_offset).min(total_visual_rows.saturating_sub(1));

    (total_visual_rows, top_visual_row)
}

/// Shared no-wrap horizontal geometry for painting, scrollbar rendering, and
/// cached mouse bounds. `content_width` includes the one-column EOL caret.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HorizontalScrollMetrics {
    pub content_width: usize,
    pub visible_text_width: usize,
    pub max_scroll: usize,
}

impl HorizontalScrollMetrics {
    pub fn new(max_visual_line_width: usize, visible_text_width: usize) -> Self {
        let content_width = max_visual_line_width.saturating_add(1);
        Self {
            content_width,
            visible_text_width,
            max_scroll: content_width.saturating_sub(visible_text_width),
        }
    }

    /// Combine the persistent source-derived maximum with this frame's
    /// canonical post-transform width. Canonical/inlay expansion deliberately
    /// remains frame-local so removing it immediately restores source bounds.
    pub fn for_frame(
        source_max_visual_width: usize,
        canonical_max_visual_width: usize,
        visible_text_width: usize,
    ) -> Self {
        Self::new(
            source_max_visual_width.max(canonical_max_visual_width),
            visible_text_width,
        )
    }
}

/// Update and return the viewport's source-derived maximum in display columns.
///
/// Only scans the currently visible lines plus a small margin. The maximum is
/// deliberately monotonic between the existing viewport reset/invalidation
/// points: a wide source line remains scrollable after it leaves the frame.
/// Post-transform `ViewLine` widths are intentionally not stored here.
pub(super) fn compute_max_line_length(
    state: &mut EditorState,
    viewport: &mut Viewport,
    tab_size: usize,
) -> usize {
    if state.buffer.is_empty() {
        return viewport.max_line_length_seen;
    }

    let visible_lines = viewport.height as usize + 5;
    let mut lines_scanned = 0usize;
    let mut iter = state.buffer.line_iterator(viewport.top_byte, 80);
    loop {
        if lines_scanned >= visible_lines {
            break;
        }
        match iter.next_line() {
            Some((_byte_offset, content)) => {
                // LineIterator includes the physical terminator. Exclude LF and
                // its optional CR partner explicitly before adding the shared
                // logical EOL-caret cell in HorizontalScrollMetrics.
                let text = content.strip_suffix('\n').unwrap_or(&content);
                let text = text.strip_suffix('\r').unwrap_or(text);
                let display_len = visual_width_with_tab_size(text, 0, tab_size);
                viewport.max_line_length_seen = viewport.max_line_length_seen.max(display_len);
                lines_scanned += 1;
            }
            None => break,
        }
    }

    viewport.max_line_length_seen
}

/// Resolve a marker's colour spec against the live theme.
///
/// Theme keys are resolved here, at paint time, rather than when the plugin
/// sets the marker — so markers follow a theme switch with no invalidation.
fn resolve_marker_color(spec: &fresh_core::api::OverlayColorSpec, theme: &Theme) -> Color {
    match spec {
        fresh_core::api::OverlayColorSpec::Rgb(r, g, b) => Color::Rgb(*r, *g, *b),
        fresh_core::api::OverlayColorSpec::ThemeKey(key) => {
            crate::view::theme::named_color_from_str(key)
                .or_else(|| theme.resolve_theme_key(key))
                .unwrap_or(Color::Reset)
        }
    }
}

/// Render a scrollbar for a split.
/// Returns (thumb_start, thumb_end) positions for mouse hit testing.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_scrollbar(
    buf: &mut ratatui::buffer::Buffer,
    state: &EditorState,
    viewport: &Viewport,
    scrollbar_rect: Rect,
    _is_active: bool,
    theme: &Theme,
    large_file_threshold_bytes: u64,
    total_lines: usize,
    top_line: usize,
    markers: &[Option<MarkerCell>],
) -> (usize, usize) {
    let height = scrollbar_rect.height as usize;
    if height == 0 {
        return (0, 0);
    }

    let buffer_len = state.buffer.len();
    let viewport_top = viewport.top_byte;
    let viewport_height_lines = height;

    let (thumb_start, thumb_size) = if buffer_len > large_file_threshold_bytes as usize {
        let thumb_start = if buffer_len > 0 {
            ((viewport_top as f64 / buffer_len as f64) * height as f64) as usize
        } else {
            0
        };
        (thumb_start, 1)
    } else {
        let thumb_size_raw = if total_lines > 0 {
            ((viewport_height_lines as f64 / total_lines as f64) * height as f64).ceil() as usize
        } else {
            1
        };

        let max_scroll_line = total_lines.saturating_sub(viewport_height_lines);

        let thumb_size = if max_scroll_line == 0 {
            height
        } else {
            let max_thumb_size = (height as f64 * 0.8).floor() as usize;
            thumb_size_raw.max(1).min(max_thumb_size).min(height)
        };

        let thumb_start = if max_scroll_line > 0 {
            let scroll_ratio = top_line.min(max_scroll_line) as f64 / max_scroll_line as f64;
            let max_thumb_start = height.saturating_sub(thumb_size);
            (scroll_ratio * max_thumb_start as f64) as usize
        } else {
            0
        };

        (thumb_start, thumb_size)
    };

    let thumb_end = thumb_start + thumb_size;

    let track_color = theme.scrollbar_track_fg;
    let thumb_color = theme.scrollbar_thumb_fg;

    for row in 0..height {
        let cell_area = Rect::new(scrollbar_rect.x, scrollbar_rect.y + row as u16, 1, 1);

        let bg = if row >= thumb_start && row < thumb_end {
            thumb_color
        } else {
            track_color
        };

        // A marker paints a half-block glyph in its colour over the track or
        // thumb background. The scrollbar is a single column
        // (`split_rendering::layout`), so a solid fill would have to choose
        // between showing the mark and showing the scroll position; the half
        // block shows both in the same cell.
        let paragraph = match markers.get(row).and_then(|m| m.as_ref()) {
            Some(marker) => Paragraph::new(MARKER_GLYPH).style(
                Style::default()
                    .fg(resolve_marker_color(&marker.color, theme))
                    .bg(bg),
            ),
            None => Paragraph::new(" ").style(Style::default().bg(bg)),
        };
        paragraph.render(cell_area, buf);
    }

    (thumb_start, thumb_end)
}

/// Glyph used for a plugin scrollbar marker: a left half block, so the
/// marker colour and the underlying track/thumb background are both visible.
pub(super) const MARKER_GLYPH: &str = "▌";

/// Render a horizontal scrollbar for a split.
/// Returns (thumb_start_col, thumb_end_col) for mouse hit testing.
pub(super) fn render_horizontal_scrollbar(
    buf: &mut ratatui::buffer::Buffer,
    viewport: &Viewport,
    hscrollbar_rect: Rect,
    _is_active: bool,
    theme: &Theme,
    metrics: HorizontalScrollMetrics,
) -> (usize, usize) {
    let width = hscrollbar_rect.width as usize;
    if width == 0 || hscrollbar_rect.height == 0 {
        return (0, 0);
    }

    let track_color = theme.scrollbar_track_fg;

    if viewport.line_wrap_enabled {
        for col in 0..width {
            let cell_area = Rect::new(hscrollbar_rect.x + col as u16, hscrollbar_rect.y, 1, 1);
            let paragraph = Paragraph::new(" ").style(Style::default().bg(track_color));
            paragraph.render(cell_area, buf);
        }
        return (0, width);
    }

    let left_column = viewport.left_column;
    let max_scroll = metrics.max_scroll;

    let (thumb_start, thumb_size) = if max_scroll == 0 {
        (0, width)
    } else {
        let thumb_size_raw = ((metrics.visible_text_width as f64 / metrics.content_width as f64)
            * width as f64)
            .ceil() as usize;
        let thumb_size = thumb_size_raw.max(2).min(width);

        let scroll_ratio = left_column.min(max_scroll) as f64 / max_scroll as f64;
        let max_thumb_start = width.saturating_sub(thumb_size);
        let thumb_start = (scroll_ratio * max_thumb_start as f64).round() as usize;

        (thumb_start, thumb_size)
    };

    let thumb_end = thumb_start + thumb_size;

    let thumb_color = theme.scrollbar_thumb_fg;

    for col in 0..width {
        let cell_area = Rect::new(hscrollbar_rect.x + col as u16, hscrollbar_rect.y, 1, 1);

        let style = if col >= thumb_start && col < thumb_end {
            Style::default().bg(thumb_color)
        } else {
            Style::default().bg(track_color)
        };

        let paragraph = Paragraph::new(" ").style(style);
        paragraph.render(cell_area, buf);
    }

    (thumb_start, thumb_end)
}

/// Render a scrollbar for composite buffer views.
pub(super) fn render_composite_scrollbar(
    buf: &mut ratatui::buffer::Buffer,
    scrollbar_rect: Rect,
    total_rows: usize,
    scroll_row: usize,
    viewport_height: usize,
    _is_active: bool,
    theme: &Theme,
) -> (usize, usize) {
    let height = scrollbar_rect.height as usize;
    if height == 0 || total_rows == 0 {
        return (0, 0);
    }

    let thumb_size_raw = if total_rows > 0 {
        ((viewport_height as f64 / total_rows as f64) * height as f64).ceil() as usize
    } else {
        1
    };

    let max_scroll = total_rows.saturating_sub(viewport_height);

    let thumb_size = if max_scroll == 0 {
        height
    } else {
        let max_thumb_size = (height as f64 * 0.8).floor() as usize;
        thumb_size_raw.max(1).min(max_thumb_size).min(height)
    };

    let thumb_start = if max_scroll > 0 {
        let scroll_ratio = scroll_row.min(max_scroll) as f64 / max_scroll as f64;
        let max_thumb_start = height.saturating_sub(thumb_size);
        (scroll_ratio * max_thumb_start as f64) as usize
    } else {
        0
    };

    let thumb_end = thumb_start + thumb_size;

    let track_color = theme.scrollbar_track_fg;
    let thumb_color = theme.scrollbar_thumb_fg;

    for row in 0..height {
        let cell_area = Rect::new(scrollbar_rect.x, scrollbar_rect.y + row as u16, 1, 1);

        let style = if row >= thumb_start && row < thumb_end {
            Style::default().bg(thumb_color)
        } else {
            Style::default().bg(track_color)
        };

        let paragraph = Paragraph::new(" ").style(style);
        paragraph.render(cell_area, buf);
    }

    (thumb_start, thumb_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn state_with_wrapping_lines(n: usize) -> EditorState {
        let fs: Arc<dyn crate::model::filesystem::FileSystem + Send + Sync> =
            Arc::new(crate::model::filesystem::StdFileSystem);
        let mut state = EditorState::new(
            80,
            24,
            crate::config::LARGE_FILE_THRESHOLD_BYTES as usize,
            fs,
        );
        // Each line is long enough to wrap to several visual rows at width 40.
        let line = "the quick brown fox jumps over the lazy dog and keeps going\n";
        let mut text = String::with_capacity(n * line.len());
        for _ in 0..n {
            text.push_str(line);
        }
        state.buffer.insert(0, &text);
        state
    }

    fn narrow_wrapped_viewport() -> Viewport {
        let mut vp = Viewport::new(40, 24);
        vp.line_wrap_enabled = true;
        vp
    }

    #[test]
    fn source_extent_uses_visual_units_and_survives_leaving_the_frame() {
        // Non-default tabs, a double-width scalar, ANSI (zero cells), and a
        // CRLF terminator all occur on the wide source line. The next frame
        // sees only a narrow line, so this catches accidental byte counts,
        // terminator counts, or replacement of the persistent source extent.
        let tab_size = 3;
        let wide = format!("{}\r\n", "a\t界\x1b[31mxe\u{301}".repeat(32));
        let narrow = "short\r\n";
        let expected = visual_width_with_tab_size(wide.trim_end_matches(['\r', '\n']), 0, tab_size);
        let mut state = state_with_wrapping_lines(0);
        state.buffer.insert(0, &(wide.clone() + narrow));
        let mut viewport = Viewport::new(80, 1);

        assert_eq!(
            compute_max_line_length(&mut state, &mut viewport, tab_size),
            expected,
            "source extent must be display cells, not source bytes"
        );
        viewport.top_byte = wide.len();
        assert_eq!(
            compute_max_line_length(&mut state, &mut viewport, tab_size),
            expected,
            "the narrow frame must retain the wide source extent"
        );
        assert_eq!(
            HorizontalScrollMetrics::new(viewport.max_line_length_seen, 0).content_width,
            expected + 1,
            "exactly one EOL caret cell is added after source visual width"
        );
    }

    #[test]
    fn canonical_inlay_extent_is_frame_local_but_source_extent_persists() {
        let source_max = 80;
        let visible = 30;
        let expanded = HorizontalScrollMetrics::for_frame(source_max, 120, visible);
        assert_eq!(
            expanded.max_scroll, 91,
            "inlay/canonical expansion may pan farther"
        );

        // Once the canonical/inlay line disappears, its transient width must
        // not leak into the viewport cache; only encountered source remains.
        let contracted = HorizontalScrollMetrics::for_frame(source_max, 0, visible);
        assert_eq!(contracted.max_scroll, 51);
        assert_eq!(contracted.content_width, source_max + 1);
    }

    #[test]
    fn horizontal_metrics_share_effective_text_width_for_compose_and_gutter() {
        // A 30-cell compose render area with a 4-cell gutter has the same
        // 26-cell text window used by paint, thumb geometry, and mouse bounds.
        let metrics = HorizontalScrollMetrics::new(40, 30usize.saturating_sub(4));
        assert_eq!(metrics.visible_text_width, 26);
        assert_eq!(metrics.max_scroll, 15);
    }

    /// Small wrapped buffers keep the exact visual-row scrollbar: total counts
    /// wrapped rows (more than logical lines) and the index is built.
    #[test]
    fn small_wrapped_buffer_uses_exact_visual_rows() {
        let mut state = state_with_wrapping_lines(100);
        let vp = narrow_wrapped_viewport();
        let buffer_len = state.buffer.len();
        let (total, _, basis) = scrollbar_line_counts(
            &mut state,
            &vp,
            crate::config::LARGE_FILE_THRESHOLD_BYTES,
            buffer_len,
        );
        assert!(
            total > 100,
            "small wrapped buffer should report wrapped-row count (>100), got {total}"
        );
        assert!(
            state.visual_row_index.line_count() > 0,
            "small wrapped buffer should build the visual-row index"
        );
        assert!(
            matches!(basis, MarkerBasis::VisualRows { .. }),
            "markers must follow the thumb onto the wrapped-row basis, got {basis:?}"
        );
    }

    /// Large wrapped buffers fall back to the logical-line approximation so the
    /// O(all-lines) visual-row scan never runs (and so never re-runs per edit):
    /// total equals the logical line count and the index is left unbuilt
    /// (fresh#2610).
    #[test]
    fn large_wrapped_buffer_skips_visual_row_scan() {
        let n = MAX_WRAP_SCROLLBAR_LINES + 1;
        let mut state = state_with_wrapping_lines(n);
        let vp = narrow_wrapped_viewport();
        let buffer_len = state.buffer.len();
        assert!(
            buffer_len <= MAX_WRAP_SCROLLBAR_BYTES,
            "test buffer should trip the line bound, not the byte bound"
        );
        let (total, _, basis) = scrollbar_line_counts(
            &mut state,
            &vp,
            crate::config::LARGE_FILE_THRESHOLD_BYTES,
            buffer_len,
        );
        assert_eq!(
            total, n,
            "large wrapped buffer should use the logical-line count, not wrapped rows"
        );
        assert_eq!(
            state.visual_row_index.line_count(),
            0,
            "large wrapped buffer must not build the O(all-lines) visual-row index"
        );
        assert!(
            matches!(basis, MarkerBasis::LogicalLines { total } if total == n as u64),
            "markers must follow the thumb onto the logical-line basis, got {basis:?}"
        );
    }

    /// The marker basis is derived in the same branch that picks the thumb's
    /// counts, so the two can never end up in different coordinate spaces.
    #[test]
    fn marker_basis_matches_the_thumb_basis_in_every_regime() {
        // Exact wrapped rows.
        let mut state = state_with_wrapping_lines(100);
        let vp = narrow_wrapped_viewport();
        let len = state.buffer.len();
        let (total, _, basis) = scrollbar_line_counts(
            &mut state,
            &vp,
            crate::config::LARGE_FILE_THRESHOLD_BYTES,
            len,
        );
        assert_eq!(
            basis,
            MarkerBasis::VisualRows {
                total: total as u64
            }
        );

        // Logical lines (wrap off).
        let mut state = state_with_wrapping_lines(100);
        let vp = Viewport::new(40, 24);
        let len = state.buffer.len();
        let (total, _, basis) = scrollbar_line_counts(
            &mut state,
            &vp,
            crate::config::LARGE_FILE_THRESHOLD_BYTES,
            len,
        );
        assert_eq!(
            basis,
            MarkerBasis::LogicalLines {
                total: total as u64
            }
        );

        // Bytes: over the large-file threshold, where no line coordinate is
        // guaranteed to exist. A tiny threshold stands in for a huge file.
        let mut state = state_with_wrapping_lines(100);
        let len = state.buffer.len();
        let (total, top, basis) = scrollbar_line_counts(&mut state, &vp, 16, len);
        assert_eq!((total, top), (0, 0), "large files use the constant thumb");
        assert_eq!(basis, MarkerBasis::Bytes { total: len as u64 });
    }

    /// The byte regime must not consult line APIs at all — that is what lets
    /// markers render on the first frame of a huge, unscanned file.
    #[test]
    fn byte_regime_projection_does_not_build_the_visual_row_index() {
        use crate::view::scrollbar_marker::ResolvedMarker;

        let mut state = state_with_wrapping_lines(100);
        let len = state.buffer.len();
        state.scrollbar_markers.set_markers(
            "test",
            vec![ResolvedMarker {
                start: len / 2,
                end: None,
                color: fresh_core::api::OverlayColorSpec::Rgb(1, 2, 3),
                priority: 0,
            }],
        );

        let cells =
            project_scrollbar_markers(&mut state, MarkerBasis::Bytes { total: len as u64 }, 20);
        assert!(
            cells.iter().any(|c| c.is_some()),
            "the marker should land somewhere on the track"
        );
        assert_eq!(
            state.visual_row_index.line_count(),
            0,
            "byte-basis projection must not touch the visual-row index"
        );
    }

    /// A buffer with no markers must not allocate or project anything.
    #[test]
    fn no_markers_means_no_projection_work() {
        let mut state = state_with_wrapping_lines(10);
        let cells = project_scrollbar_markers(&mut state, MarkerBasis::Bytes { total: 100 }, 20);
        assert!(cells.is_empty());
    }
}
