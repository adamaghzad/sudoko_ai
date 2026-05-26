"use client";
import { motion } from "framer-motion";
import { TrendingDown, TrendingUp } from "lucide-react";
import type { TrainingHistory } from "@/lib/types";

interface TrainingChartProps {
  history: TrainingHistory;
  color: string;
  modelName: string;
}

// Minimal SVG line chart — no external chart lib needed
function LineChart({
  series,
  height = 80,
  colors,
}: {
  series: { label: string; data: number[]; color: string }[];
  height?: number;
  colors?: string[];
}) {
  const maxLen = Math.max(...series.map((s) => s.data.length));
  if (maxLen === 0) return null;

  const allVals = series.flatMap((s) => s.data).filter(isFinite);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;

  const W = 400;
  const H = height;
  const pad = { t: 8, r: 8, b: 20, l: 36 };
  const chartW = W - pad.l - pad.r;
  const chartH = H - pad.t - pad.b;

  const toX = (i: number, n: number) => pad.l + (i / Math.max(n - 1, 1)) * chartW;
  const toY = (v: number) => pad.t + chartH - ((v - minV) / range) * chartH;

  const makePath = (data: number[]) =>
    data
      .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i, data.length).toFixed(1)} ${toY(v).toFixed(1)}`)
      .join(" ");

  // Y axis labels
  const yTicks = [minV, (minV + maxV) / 2, maxV].map((v) => ({
    v,
    y: toY(v),
    label: v < 1 ? v.toFixed(3) : v.toFixed(2),
  }));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      {/* Grid lines */}
      {yTicks.map(({ y, label }, i) => (
        <g key={i}>
          <line x1={pad.l} y1={y} x2={W - pad.r} y2={y}
                stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
          <text x={pad.l - 4} y={y + 4} textAnchor="end"
                fill="rgba(255,255,255,0.3)" fontSize={9}>{label}</text>
        </g>
      ))}
      {/* X label */}
      <text x={W / 2} y={H - 2} textAnchor="middle"
            fill="rgba(255,255,255,0.3)" fontSize={9}>Époques</text>

      {/* Lines */}
      {series.map((s, si) => (
        <g key={si}>
          <path d={makePath(s.data)} fill="none"
                stroke={s.color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
                opacity={0.9} />
          {/* Last point dot */}
          {s.data.length > 0 && (
            <circle
              cx={toX(s.data.length - 1, s.data.length)}
              cy={toY(s.data[s.data.length - 1])}
              r={3} fill={s.color}
            />
          )}
        </g>
      ))}
    </svg>
  );
}

export default function TrainingChart({ history, color, modelName }: TrainingChartProps) {
  const finalTrainLoss = history.train_loss.at(-1) ?? 0;
  const finalValLoss = history.val_loss.at(-1) ?? 0;
  const finalPuzzleAcc = history.val_puzzle_acc.at(-1) ?? 0;
  const finalCellAcc = history.val_cell_acc.at(-1) ?? 0;
  const finalEmptyCellAcc = history.val_empty_cell_acc?.at(-1) ?? null;

  const lossSeries = [
    { label: "Train Loss", data: history.train_loss, color },
    { label: "Val Loss",   data: history.val_loss,   color: "#ef4444" },
  ];
  const accSeries = [
    { label: "Cell Acc",        data: history.val_cell_acc,        color: "#10b981" },
    { label: "Puzzle Acc",      data: history.val_puzzle_acc,       color: "#f59e0b" },
    ...(history.val_empty_cell_acc?.length
      ? [{ label: "Vides Acc", data: history.val_empty_cell_acc, color: "#a78bfa" }]
      : []),
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl p-4 space-y-3"
      style={{ background: `${color}08`, border: `1px solid ${color}20` }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-widest" style={{ color }}>
          {modelName}
        </span>
        <span className="text-xs text-slate-500">{history.epochs_trained} époques</span>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <StatBadge label="Val Loss" value={finalValLoss.toFixed(4)}
                   icon={<TrendingDown className="w-3 h-3" />} color="#ef4444" />
        <StatBadge label="Puzzle Acc" value={`${(finalPuzzleAcc * 100).toFixed(1)}%`}
                   icon={<TrendingUp className="w-3 h-3" />} color="#f59e0b" />
        <StatBadge label="Cell Acc" value={`${(finalCellAcc * 100).toFixed(1)}%`}
                   icon={<TrendingUp className="w-3 h-3" />} color="#10b981" />
        <StatBadge label="Train Loss" value={finalTrainLoss.toFixed(4)}
                   icon={<TrendingDown className="w-3 h-3" />} color={color} />
        {finalEmptyCellAcc !== null && (
          <StatBadge label="Vides Acc" value={`${(finalEmptyCellAcc * 100).toFixed(1)}%`}
                     icon={<TrendingUp className="w-3 h-3" />} color="#a78bfa" />
        )}
        {history.best_epoch !== undefined && (
          <StatBadge label="Meill. époque" value={`#${history.best_epoch}`}
                     icon={<TrendingUp className="w-3 h-3" />} color="#94a3b8" />
        )}
        {history.best_val_cell_acc !== undefined && (
          <StatBadge label="Meill. Cell" value={`${(history.best_val_cell_acc * 100).toFixed(1)}%`}
                     icon={<TrendingUp className="w-3 h-3" />} color="#10b981" />
        )}
        {history.train_time_sec !== undefined && (
          <StatBadge label="Durée" value={`${history.train_time_sec}s`}
                     icon={<TrendingDown className="w-3 h-3" />} color="#64748b" />
        )}
      </div>

      {/* Loss curves */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <Legend color={color} label="Train Loss" />
          <Legend color="#ef4444" label="Val Loss" />
        </div>
        <div className="bg-black/30 rounded-lg overflow-hidden p-1">
          <LineChart series={lossSeries} height={80} />
        </div>
      </div>

      {/* Accuracy curves */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <Legend color="#10b981" label="Cell Acc" />
          <Legend color="#f59e0b" label="Puzzle Acc" />
          {history.val_empty_cell_acc?.length && <Legend color="#a78bfa" label="Vides Acc" />}
        </div>
        <div className="bg-black/30 rounded-lg overflow-hidden p-1">
          <LineChart series={accSeries} height={80} />
        </div>
      </div>
    </motion.div>
  );
}

function StatBadge({ label, value, icon, color }: {
  label: string; value: string; icon: React.ReactNode; color: string;
}) {
  return (
    <div className="bg-black/30 rounded-lg px-2 py-1.5 flex items-center gap-1.5">
      <span style={{ color }}>{icon}</span>
      <span className="text-slate-500">{label}:</span>
      <span className="font-bold" style={{ color }}>{value}</span>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <div className="w-3 h-0.5 rounded" style={{ background: color }} />
      <span className="text-[10px] text-slate-500">{label}</span>
    </div>
  );
}
