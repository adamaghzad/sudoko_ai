"use client";
import { motion } from "framer-motion";
import clsx from "clsx";

interface SudokuGridProps {
  puzzle: number[];
  solution?: number[] | null;
  highlightSolved?: boolean;
  size?: "sm" | "md" | "lg";
  label?: string;
  color?: string;
}

const SIZE_CLASSES = {
  sm: { cell: "w-7 h-7 text-xs", grid: "gap-[1px]" },
  md: { cell: "w-9 h-9 text-sm", grid: "gap-[1px]" },
  lg: { cell: "w-11 h-11 text-base", grid: "gap-[2px]" },
};

export default function SudokuGrid({
  puzzle,
  solution,
  highlightSolved = true,
  size = "md",
  label,
  color = "#6366f1",
}: SudokuGridProps) {
  const { cell: cellCls, grid: gridCls } = SIZE_CLASSES[size];
  const display = solution ?? puzzle;

  return (
    <div className="flex flex-col items-center gap-2">
      {label && (
        <span className="text-xs font-semibold uppercase tracking-widest"
              style={{ color }}>
          {label}
        </span>
      )}
      <div
        className={clsx("grid grid-cols-9", gridCls)}
        style={{
          border: `2px solid ${color}80`,
          borderRadius: 8,
          padding: 2,
          background: "rgba(0,0,0,0.3)",
        }}
      >
        {display.map((val, idx) => {
          const row = Math.floor(idx / 9);
          const col = idx % 9;
          const isClue = puzzle[idx] !== 0;
          const isSolved = !isClue && val !== 0 && solution !== null;
          const isEmpty = val === 0;

          const borderLeft = col % 3 === 0 && col !== 0 ? `2px solid ${color}80` : undefined;
          const borderTop = row % 3 === 0 && row !== 0 ? `2px solid ${color}80` : undefined;

          return (
            <motion.div
              key={idx}
              className={clsx(
                cellCls,
                "flex items-center justify-center font-bold select-none rounded-[2px]",
                isClue && "text-indigo-200",
                isSolved && highlightSolved && "text-emerald-400",
                isEmpty && "text-slate-600",
              )}
              style={{
                borderLeft,
                borderTop,
                background: isClue
                  ? "rgba(99,102,241,0.08)"
                  : isSolved
                  ? "rgba(16,185,129,0.06)"
                  : "rgba(255,255,255,0.02)",
              }}
              initial={isSolved ? { scale: 0.5, opacity: 0 } : false}
              animate={isSolved ? { scale: 1, opacity: 1 } : {}}
              transition={{ delay: idx * 0.003, type: "spring", stiffness: 300 }}
            >
              {isEmpty ? "" : val}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
