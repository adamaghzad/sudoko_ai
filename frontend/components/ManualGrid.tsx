"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import clsx from "clsx";

interface ManualGridProps {
  puzzle: number[];
  onChange: (puzzle: number[]) => void;
  disabled?: boolean;
}

export default function ManualGrid({ puzzle, onChange, disabled }: ManualGridProps) {
  const [focused, setFocused] = useState<number | null>(null);

  const handleKey = (idx: number, e: React.KeyboardEvent<HTMLInputElement>) => {
    const key = e.key;
    if (key >= "1" && key <= "9") {
      const next = [...puzzle];
      next[idx] = parseInt(key);
      onChange(next);
    } else if (key === "Backspace" || key === "Delete" || key === "0") {
      const next = [...puzzle];
      next[idx] = 0;
      onChange(next);
    } else if (key === "ArrowRight" && idx < 80) setFocused(idx + 1);
    else if (key === "ArrowLeft" && idx > 0) setFocused(idx - 1);
    else if (key === "ArrowDown" && idx + 9 <= 80) setFocused(idx + 9);
    else if (key === "ArrowUp" && idx - 9 >= 0) setFocused(idx - 9);
  };

  return (
    <div
      className="inline-grid grid-cols-9"
      style={{
        border: "2px solid rgba(99,102,241,0.5)",
        borderRadius: 10,
        padding: 3,
        background: "rgba(0,0,0,0.4)",
        gap: 2,
      }}
    >
      {puzzle.map((val, idx) => {
        const row = Math.floor(idx / 9);
        const col = idx % 9;
        const isFocused = focused === idx;
        const borderLeft = col % 3 === 0 && col !== 0 ? "2px solid rgba(99,102,241,0.6)" : undefined;
        const borderTop = row % 3 === 0 && row !== 0 ? "2px solid rgba(99,102,241,0.6)" : undefined;

        return (
          <motion.input
            key={idx}
            type="text"
            inputMode="numeric"
            maxLength={1}
            value={val === 0 ? "" : val}
            readOnly={disabled}
            onFocus={() => setFocused(idx)}
            onBlur={() => setFocused(null)}
            onKeyDown={(e) => handleKey(idx, e)}
            onChange={() => {}}
            whileFocus={{ scale: 1.05 }}
            className={clsx(
              "w-10 h-10 text-center text-sm font-bold outline-none rounded-[3px]",
              "transition-all duration-150 caret-transparent",
              isFocused
                ? "text-white ring-1 ring-indigo-400"
                : val !== 0
                ? "text-indigo-200"
                : "text-slate-500",
              disabled && "cursor-default"
            )}
            style={{
              borderLeft,
              borderTop,
              background: isFocused ? "rgba(99,102,241,0.2)" : undefined,
            }}
          />
        );
      })}
    </div>
  );
}
