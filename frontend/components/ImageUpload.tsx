"use client";
import { useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImageIcon, X, Loader2 } from "lucide-react";
import clsx from "clsx";

interface ImageUploadProps {
  onSolve: (file: File) => Promise<void>;
  loading: boolean;
}

export default function ImageUpload({ onSolve, loading }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clear = () => {
    setPreview(null);
    setSelectedFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="flex flex-col gap-4">
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={clsx(
              "drop-zone h-52 rounded-2xl",
              dragging && "active"
            )}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <motion.div
              animate={{ y: dragging ? -8 : 0 }}
              className="flex flex-col items-center gap-3 text-slate-400"
            >
              <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/30
                              flex items-center justify-center">
                <ImageIcon className="w-8 h-8 text-indigo-400" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-slate-300">
                  Déposez une photo de Sudoku
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  ou cliquez pour parcourir (JPG, PNG)
                </p>
              </div>
            </motion.div>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="relative rounded-2xl overflow-hidden border border-indigo-500/30"
          >
            <img
              src={preview}
              alt="Sudoku preview"
              className="w-full h-52 object-contain bg-black/40"
            />
            <button
              onClick={clear}
              className="absolute top-2 right-2 w-8 h-8 rounded-full bg-slate-800/80
                         flex items-center justify-center hover:bg-red-900/80 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        onClick={() => selectedFile && onSolve(selectedFile)}
        disabled={!selectedFile || loading}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={clsx(
          "w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200",
          selectedFile && !loading
            ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/25"
            : "bg-slate-800 text-slate-500 cursor-not-allowed"
        )}
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyse en cours...
          </>
        ) : (
          <>
            <Upload className="w-4 h-4" />
            Analyser et Résoudre
          </>
        )}
      </motion.button>
    </div>
  );
}
