"""
Pipeline de traitement d'image Sudoku.
Étapes : Détection grille → Transformation perspective → Extraction cellules → Prétraitement
"""
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points : [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transformation perspective vers vue de face."""
    rect = order_points(pts)
    tl, tr, br, bl = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    side = max(maxW, maxH)  # carré
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (side, side))


def detect_grid(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Détecte et extrait la grille Sudoku par analyse de contours.
    Retourne la grille redressée ou None si non trouvée.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Plus grand contour = grille Sudoku
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < 10_000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4, 2))

    return None


def extract_cells(grid_image: np.ndarray, cell_size: int = 28) -> np.ndarray:
    """
    Découpe la grille en 81 cellules normalisées (cell_size × cell_size, niveaux de gris).
    Retourne un array (81, 1, cell_size, cell_size) float32 ∈ [0, 1].
    """
    gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY) \
        if len(grid_image.shape) == 3 else grid_image

    h, w = gray.shape
    cell_h, cell_w = h // 9, w // 9
    margin = 0.1  # rognage des bords pour éviter les lignes de grille

    cells = []
    for row in range(9):
        for col in range(9):
            y1 = int(row * cell_h + cell_h * margin)
            y2 = int((row + 1) * cell_h - cell_h * margin)
            x1 = int(col * cell_w + cell_w * margin)
            x2 = int((col + 1) * cell_w - cell_w * margin)
            cell = gray[y1:y2, x1:x2]
            cell = cv2.resize(cell, (cell_size, cell_size))
            # Binarisation pour isoler le chiffre
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cell = cell.astype(np.float32) / 255.0
            cells.append(cell)

    return np.stack(cells).reshape(81, 1, cell_size, cell_size)  # (81, 1, H, W)


def is_empty_cell(cell: np.ndarray, threshold: float = 0.98) -> bool:
    """Cellule vide si presque entièrement blanche (peu de pixels noirs)."""
    white_ratio = cell.mean()
    return white_ratio > threshold


def preprocess_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Charge une image depuis des bytes et la retourne en BGR OpenCV."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_grid(image_bytes: bytes, cell_size: int = 28) -> dict:
    """
    Pipeline complet : bytes → grille détectée → cellules extraites.
    Retourne un dict avec la grille et les cellules prétraitées.
    """
    img = preprocess_image_bytes(image_bytes)
    if img is None:
        raise ValueError("Impossible de décoder l'image")

    grid_img = detect_grid(img)
    if grid_img is None:
        raise ValueError("Grille Sudoku non détectée dans l'image")

    cells = extract_cells(grid_img, cell_size)
    empty_flags = [is_empty_cell(cells[i, 0]) for i in range(81)]

    # Encode la grille détectée en JPEG pour preview
    _, buf = cv2.imencode(".jpg", cv2.resize(grid_img, (450, 450)))
    grid_b64 = buf.tobytes()

    return {
        "cells": cells,            # (81, 1, 28, 28)
        "empty_flags": empty_flags,
        "grid_image_bytes": grid_b64,
    }
