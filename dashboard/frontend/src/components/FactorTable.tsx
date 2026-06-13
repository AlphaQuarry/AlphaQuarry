import {
  type ColumnDef,
  type OnChangeFn,
  type SortingState,
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import { BarChart3, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { FactorMetric } from "../types";
import { formatNumber, formatPercent, formatPermille } from "../utils/format";

const DASHBOARD_COLUMNS: Array<ColumnDef<FactorMetric>> = [
  { accessorKey: "factor", header: "Name", size: 180 },
  { accessorKey: "effectiveness_tier", header: "Tier", size: 70 },
  { accessorKey: "period", header: "Period", size: 70 },
  { accessorKey: "layers", header: "Layers", size: 70 },
  { accessorKey: "ic_mean", header: "IC", cell: (info) => formatNumber(info.getValue<number | null>(), 4), size: 90 },
  { accessorKey: "ir", header: "IR", cell: (info) => formatNumber(info.getValue<number | null>(), 2), size: 90 },
  {
    accessorKey: "long_short_annualized_return",
    header: "LS Return",
    cell: (info) => formatPercent(info.getValue<number | null>()),
    size: 110
  },
  {
    accessorKey: "long_short_sharpe_ratio",
    header: "LS Sharpe",
    cell: (info) => formatNumber(info.getValue<number | null>(), 2),
    size: 105
  },
  {
    accessorKey: "best_layer_annualized_return",
    header: "LO Return",
    cell: (info) => formatPercent(info.getValue<number | null>()),
    size: 110
  },
  {
    accessorKey: "best_layer_sharpe",
    header: "LO Sharpe",
    cell: (info) => formatNumber(info.getValue<number | null>(), 2),
    size: 105
  },
  {
    accessorKey: "turnover_long_only_mean",
    header: "Turnover",
    cell: (info) => formatPercent(info.getValue<number | null>()),
    size: 110
  },
  {
    accessorKey: "margin_long_only",
    header: "Margin",
    cell: (info) => formatPermille(info.getValue<number | null>()),
    size: 105
  },
  {
    accessorKey: "feedback_score",
    header: "Feedback Score",
    cell: (info) => formatNumber(info.getValue<number | null>(), 1),
    size: 125
  },
  { accessorKey: "score_total", header: "Full Score", cell: (info) => formatNumber(info.getValue<number | null>(), 1), size: 105 }
];

type FactorTableProps = {
  factors: FactorMetric[];
  total: number;
  loading: boolean;
  sorting: SortingState;
  selectedFactor: FactorMetric | null;
  runStatus: string | null;
  effectiveOnly: boolean;
  pageIndex: number;
  pageSize: number;
  onSortingChange: OnChangeFn<SortingState>;
  onSelectFactor: (factor: FactorMetric) => void;
  onPageChange: (pageIndex: number) => void;
  onPageSizeChange: (pageSize: number) => void;
};

export function FactorTable({
  factors,
  total,
  loading,
  sorting,
  selectedFactor,
  runStatus,
  effectiveOnly,
  pageIndex,
  pageSize,
  onSortingChange,
  onSelectFactor,
  onPageChange,
  onPageSizeChange
}: FactorTableProps) {
  const [customSize, setCustomSize] = useState("");
  const table = useReactTable({
    data: factors,
    columns: DASHBOARD_COLUMNS,
    state: { sorting },
    manualSorting: true,
    onSortingChange,
    getCoreRowModel: getCoreRowModel()
  });
  const colSpan = DASHBOARD_COLUMNS.length + 1;
  const totalPages = Math.max(1, Math.ceil(total / Math.max(1, pageSize)));
  const startRow = total === 0 ? 0 : pageIndex * pageSize + 1;
  const endRow = total === 0 ? 0 : Math.min(total, pageIndex * pageSize + factors.length);
  const canPrev = pageIndex > 0;
  const canNext = pageIndex + 1 < totalPages;

  const applyPageSize = (value: number) => {
    const next = Math.max(1, Math.min(5000, Math.floor(value)));
    onPageSizeChange(next);
    onPageChange(0);
  };

  return (
    <section className="table-panel">
      <div className="table-status">
        <span>{total.toLocaleString()} factors</span>
        <span>{runStatus}</span>
      </div>
      <div className="table-scroll">
        <table className="factor-table">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                <th className="utility-head" aria-label="row actions">
                  <SlidersHorizontal size={16} />
                </th>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} style={{ width: header.getSize() }}>
                    <button type="button" onClick={header.column.getToggleSortingHandler()} className="header-sort">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      <SortIcon direction={header.column.getIsSorted()} />
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={colSpan} className="empty-cell">Loading</td>
              </tr>
            ) : null}
            {!loading && factors.length === 0 ? (
              <tr>
                <td colSpan={colSpan} className="empty-cell">
                  {effectiveOnly
                    ? "No effective factors yet. Current filters require S/A/B tiers or score >= 60; continue closed-loop mining or adjust parameters."
                    : "No factors available. Run a closed-loop job or refresh local artifacts."}
                </td>
              </tr>
            ) : null}
            {!loading && table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={selectedFactor?.factor === row.original.factor ? "selected-row" : ""}
                onClick={() => onSelectFactor(row.original)}
              >
                <td className="row-actions"><BarChart3 size={16} /></td>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className={cell.column.id === "factor" ? "factor-name" : ""}>
                    {cell.column.id === "effectiveness_tier" ? (
                      <span className={`tier tier-${String(cell.getValue() ?? "").toLowerCase()}`}>
                        {String(cell.getValue() ?? "-")}
                      </span>
                    ) : (
                      cell.column.columnDef.cell
                        ? flexRender(cell.column.columnDef.cell, cell.getContext())
                        : String(cell.getValue() ?? "")
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-footer">
        <span>
          Showing {startRow.toLocaleString()}-{endRow.toLocaleString()} of {total.toLocaleString()}
        </span>
        <div className="pagination-controls">
          <button type="button" disabled={loading || !canPrev} onClick={() => onPageChange(pageIndex - 1)}>
            Prev
          </button>
          <strong>Page {Math.min(pageIndex + 1, totalPages)} / {totalPages}</strong>
          <button type="button" disabled={loading || !canNext} onClick={() => onPageChange(pageIndex + 1)}>
            Next
          </button>
          <label>
            Page size
            <select value={[10, 20, 50].includes(pageSize) ? String(pageSize) : "custom"} onChange={(event) => {
              const value = event.target.value;
              if (value === "custom") {
                return;
              }
              applyPageSize(Number(value));
            }}>
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="custom">Custom</option>
            </select>
          </label>
          <input
            type="number"
            min={1}
            max={5000}
            value={customSize}
            placeholder={String(pageSize)}
            onChange={(event) => setCustomSize(event.target.value)}
            onBlur={() => {
              if (customSize.trim()) {
                applyPageSize(Number(customSize));
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && customSize.trim()) {
                applyPageSize(Number(customSize));
              }
            }}
            aria-label="Custom page size"
          />
        </div>
      </div>
    </section>
  );
}

function SortIcon({ direction }: { direction: false | "asc" | "desc" }) {
  if (direction === "asc") {
    return <span className="sort-arrow" aria-label="sorted ascending">↑</span>;
  }
  if (direction === "desc") {
    return <span className="sort-arrow" aria-label="sorted descending">↓</span>;
  }
  return null;
}
