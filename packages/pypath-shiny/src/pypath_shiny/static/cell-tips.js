/* A tooltip on every table cell: "<row label> · <column>: <value>".
 *
 * One delegated Bootstrap tooltip covers the three kinds of table on the
 * pages: plain tables (.cell-table), Shiny data grids and great_tables
 * tables. The value is the cell's text unless the server gave more:
 *   - data-full / data-remark on the cell (plain tables), or
 *   - a <script class="cell-tips-data" data-for="OUTPUT_ID"> JSON block
 *     {rowLabel: {column: {full, remark}}} beside a grid.
 * See services/cell_tables.py.
 */
(function () {
  "use strict";

  var CELLS =
    ".cell-table tbody td, .shiny-data-grid tbody td, table.gt_table td.gt_row";
  var parsed = new WeakMap(); // script element -> its parsed JSON

  function text(el) {
    return el ? el.textContent.replace(/\s+/g, " ").trim() : "";
  }

  // The heading above a cell: the last heading row with a cell at its index
  // (a great_tables title or a grid's filter row is not a column label).
  function columnLabel(table, cell) {
    if (!table.tHead) return "";
    var rows = table.tHead.rows;
    for (var i = rows.length - 1; i >= 0; i--) {
      if (rows[i].cells.length <= cell.cellIndex) continue;
      var label = text(rows[i].cells[cell.cellIndex]);
      if (label) return label;
    }
    return "";
  }

  // The data for a table: a block returned with it, in its own output, or
  // one another output rendered for it (a grid's output holds only the grid).
  function lookup(table, rowLabel, column) {
    var output = table.closest(".shiny-bound-output");
    if (!output) return null;
    var block = output.querySelector("script.cell-tips-data:not([data-for])");
    if (!block && output.id) {
      block = document.querySelector(
        'script.cell-tips-data[data-for="' + CSS.escape(output.id) + '"]'
      );
    }
    if (!block) return null;
    if (!parsed.has(block)) {
      var data = {};
      try {
        data = JSON.parse(block.textContent) || {};
      } catch (e) {
        console.warn("[cell-tips] unreadable data for " + output.id);
      }
      parsed.set(block, data);
    }
    var row = parsed.get(block)[rowLabel];
    return (row && row[column]) || null;
  }

  // Whether a full value is the number the cell shows, before rounding.
  // A cell edited in a grid shows its new value before the server has sent
  // new data, and the old full value must not be passed off as it.
  function roundsTo(full, shown) {
    var text = shown.replace(/,/g, "");
    if (text === "" || isNaN(Number(text)) || isNaN(Number(full))) return false;
    var places = (text.split(".")[1] || "").replace(/[^0-9].*$/, "").length;
    return Math.abs(Number(full) - Number(text)) <= 0.5000001 * Math.pow(10, -places);
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function tip(cell) {
    var table = cell.closest("table");
    var row = cell.parentElement;
    if (!table || !row) return "";
    if (cell.querySelector("input, textarea, select")) return ""; // being edited
    var shown = text(cell);
    var column = columnLabel(table, cell);
    var rowLabel = cell.cellIndex === 0 ? "" : text(row.cells[0]);
    var extra = lookup(table, text(row.cells[0]), column) || {};
    var full = cell.getAttribute("data-full") || extra.full || "";
    if (full && !roundsTo(full, shown)) full = "";
    var remark = cell.getAttribute("data-remark") || extra.remark || "";
    var value = full || shown;
    if (!value && !remark) return "";

    var where = [rowLabel, column].filter(Boolean).join(" · ");
    var html = "";
    if (where) html += '<div class="cell-tip-where">' + escapeHtml(where) + "</div>";
    if (value) html += '<div class="cell-tip-value">' + escapeHtml(value) + "</div>";
    if (remark) html += '<div class="cell-tip-remark">' + escapeHtml(remark) + "</div>";
    return html;
  }

  function start() {
    if (!window.bootstrap || !bootstrap.Tooltip) return;
    new bootstrap.Tooltip(document.body, {
      selector: CELLS,
      title: function () {
        return tip(this);
      },
      html: true, // every piece is escaped above
      sanitize: false,
      container: "body",
      trigger: "hover",
      placement: "top",
      delay: { show: 350, hide: 0 },
      customClass: "cell-tip",
    });
    // A tooltip must not sit over a cell while it is edited or the table scrolls.
    function hideAll() {
      document.querySelectorAll(".tooltip.cell-tip").forEach(function (el) {
        el.remove();
      });
    }
    document.addEventListener("mousedown", hideAll, true);
    document.addEventListener("scroll", hideAll, true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
