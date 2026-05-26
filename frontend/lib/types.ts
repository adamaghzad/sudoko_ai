export type ModelName = "mlp" | "cnn" | "rnn" | "lstm" | "gru" | "hybrid" | "backtracking";

export interface ModelResult {
  solution: number[] | null;
  time_ms: number;
  is_valid: boolean;
  confidence: number;
  confidence_empty: number;
  confidence_min: number;
  confidence_max: number;
  trained: boolean;
  color: string;
  arch: string;
  desc: string;
}

export interface SolveResponse {
  puzzle: number[];
  num_clues: number;
  results: Record<ModelName, ModelResult>;
  puzzle_valid: boolean;
  puzzle_error: string;
  ocr?: {
    cells: Array<{ digit: number; confidence: number; is_empty: boolean }>;
    grid_image_b64: string;
  };
}

export interface ModelStatus {
  trained: boolean;
  weight_exists: boolean;
  weight_size_mb: number;
  color: string;
  arch: string;
  desc: string;
  history: TrainingHistory | null;
}

export interface TrainingHistory {
  train_loss: number[];
  val_loss: number[];
  train_cell_acc: number[];
  val_cell_acc: number[];
  val_puzzle_acc: number[];
  val_empty_cell_acc?: number[];
  epochs_trained: number;
  best_val_loss?: number;
  best_val_puzzle_acc?: number;
  best_val_cell_acc?: number;
  best_epoch?: number;
  train_time_sec?: number;
}

export interface GenerateResponse {
  puzzle: number[];
  solution: number[];
  num_clues: number;
  formatted: string;
}

export type AppMode = "manual" | "image" | "generate";
