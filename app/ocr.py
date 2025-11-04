from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR

# força 1 thread - testes mostraram q é mais rápido assim
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

# guarda instancia do OCR pra não ter q recarregar
_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_ocr() -> PaddleOCR:
    # inicializa o OCR com configs otimizadas
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=False,          # desligado pra ser mais rápido
                lang="en",
                use_gpu=False,
                show_log=False,
                
                # thresholds baixos pra pegar todos os números
                det_limit_side_len=1600,      # preserva detalhes
                det_db_thresh=0.15,           # bem sensível
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.5,      # expande as boxes
                
                # reconhecimento
                rec_batch_num=6,
                drop_score=0.2,               # aceita score baixo
                max_text_length=5,
            )
            print("PaddleOCR inicializado (máxima detecção)", flush=True)
        except Exception as e:
            print(f"Warning: Erro config PaddleOCR: {e}", flush=True)
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            print("PaddleOCR inicializado (fallback)", flush=True)
    return _OCR_INSTANCE

def enhance_image_for_digits(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    # pre-processa imagem com varias tecnicas pra melhorar detecção
    # ajusta tamanho
    h, w = image_bgr.shape[:2]
    max_dim = 1600
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w
    elif max(h, w) < 800:  # upscale se pequena
        scale = 1200 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w
    
    # converte pra LAB pq facilita mexer na luminancia
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # tecnica 1: CLAHE + denoise
    l_denoised = cv2.fastNlMeansDenoising(l, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_denoised)
    
    lab_enhanced = cv2.merge([l_clahe, a, b])
    enhanced1 = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # sharpening com unsharp mask
    gaussian = cv2.GaussianBlur(enhanced1, (0, 0), 2.0)
    enhanced1 = cv2.addWeighted(enhanced1, 2.0, gaussian, -1.0, 0)
    
    # tecnica 2: morphological
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    gray_eq = cv2.equalizeHist(gray)
    
    # gradient morfologico realça bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray_eq, cv2.MORPH_GRADIENT, kernel)
    
    enhanced2 = cv2.addWeighted(gray_eq, 0.7, morph, 0.3, 0)
    
    enhanced2 = cv2.addWeighted(enhanced2, 1.5, cv2.GaussianBlur(enhanced2, (0, 0), 1.0), -0.5, 0)
    enhanced2_bgr = cv2.cvtColor(enhanced2, cv2.COLOR_GRAY2BGR)
    
    # tecnica 3: adaptive threshold multiplo
    gray_clahe = clahe.apply(gray)
    
    # bilateral preserva bordas e remove ruido
    bilateral = cv2.bilateralFilter(gray_clahe, 9, 75, 75)
    
    adaptive1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    
    adaptive2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 15, 3)
    
    adaptive_combined = cv2.bitwise_or(adaptive1, adaptive2)
    
    # limpa ruido com morfologia
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    adaptive_clean = cv2.morphologyEx(adaptive_combined, cv2.MORPH_CLOSE, kernel_small)
    adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_OPEN, kernel_small)
    
    adaptive_bgr = cv2.cvtColor(adaptive_clean, cv2.COLOR_GRAY2BGR)
    
    # tecnica 4: contrast stretching + otsu
    norm = cv2.normalize(gray_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    _, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # inverte se o fundo for escuro
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    
    otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    
    enhanced1 = np.clip(enhanced1, 0, 255).astype(np.uint8)
    enhanced2_bgr = np.clip(enhanced2_bgr, 0, 255).astype(np.uint8)
    
    # faz upscale pra pegar numeros pequenos
    upscaled_15 = cv2.resize(enhanced1, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    upscaled_20 = cv2.resize(enhanced1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # testa 4 variantes (balanço entre velocidade e precisão)
    variants = [
        {"img": image_bgr, "sx": 1.0, "sy": 1.0},
        {"img": enhanced1, "sx": 1.0, "sy": 1.0},
        {"img": upscaled_15, "sx": 1.5, "sy": 1.5},
        {"img": upscaled_20, "sx": 2.0, "sy": 2.0},
    ]
    
    return variants


def _only_digit_sequences(text: str) -> List[str]:
    # extrai só os numeros do texto, mantendo zeros na frente
    text_clean = re.sub(r'[^\d]', '', text)
    if not text_clean:
        return []
    
    seqs = re.findall(r"\d{1,4}", text)
    cleaned: List[str] = []
    for s in seqs:
        if s.isdigit() and 1 <= len(s) <= 4:
            cleaned.append(s)  # mantem zero na frente
    
    # se nao achou, pega digito por digito
    if not cleaned and text_clean:
        for char in text_clean:
            if char.isdigit():
                cleaned.append(char)
    
    return list(set(cleaned))


def extract_jersey_numbers(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    # funcao principal - extrai os numeros das camisas
    import time
    start = time.time()
    
    ocr = get_ocr()
    
    variants = enhance_image_for_digits(image_bgr)
    prep_time = time.time() - start
    
    candidates: List[Dict[str, Any]] = []

    ocr_start = time.time()
    for v in variants:
        variant = v["img"]
        sx, sy = float(v["sx"]), float(v["sy"])
        
        try:
            result = ocr.ocr(variant, cls=False)
        except Exception as e:
            print(f"Erro OCR: {e}", flush=True)
            continue
            
        if not result:
            continue
        
        # processa cada detecção
        for line in result:
            if not line:
                continue
            for det in line or []:
                if not det or len(det) < 2:
                    continue
                    
                box = det[0]
                info = det[1]
                if box is None or info is None or len(info) < 2:
                    continue
                    
                text = info[0] if isinstance(info[0], str) else ""
                try:
                    conf = float(info[1])
                except Exception:
                    conf = 0.0
                
                digit_seqs = _only_digit_sequences(text)
                
                # se nao achou nada, tenta extrair direto
                if not digit_seqs:
                    text_digits_only = re.sub(r'[^\d]', '', text)
                    if text_digits_only:
                        if 1 <= len(text_digits_only) <= 4:
                            digit_seqs = [text_digits_only]
                
                # aceita de 1 a 4 digitos (numeros de peito)
                digit_seqs = [s for s in digit_seqs if s and 1 <= len(s) <= 4 and int(s) <= 9999]
                
                if not digit_seqs or box is None:
                    continue
                
                # reescala as coordenadas
                try:
                    xs = [p[0] / sx for p in box]
                    ys = [p[1] / sy for p in box]
                    x_min, y_min = int(min(xs)), int(min(ys))
                    x_max, y_max = int(max(xs)), int(max(ys))
                except Exception:
                    continue
                    
                w = max(1, x_max - x_min)
                h = max(1, y_max - y_min)
                
                bbox = {"x": x_min, "y": y_min, "w": w, "h": h}
                
                # adiciona cada numero encontrado
                for seq in digit_seqs:
                    if 1 <= len(seq) <= 4:
                        # ajusta confiança
                        seq_conf = conf if len(seq) == len(text.strip()) else conf * 0.80
                        candidates.append({
                            "number": seq,
                            "confidence": round(seq_conf, 4),
                            "bbox": bbox,
                        })
    
    if not candidates:
        return []

    from collections import defaultdict
    
    # agrupa por numero
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[c["number"]].append(c)

    # filtra e pega os melhores
    final_results: List[Dict[str, Any]] = []
    for num, group in grouped.items():
        try:
            num_int = int(num)
            if num_int < 0 or num_int > 9999:
                continue
        except ValueError:
            continue
        
        # pega o de maior confianca
        best = max(group, key=lambda x: x["confidence"])
        
        # define threshold minimo (bem permissivo)
        if len(num) == 1:
            min_conf = 0.30
        elif len(num) == 2:
            min_conf = 0.25
        elif len(num) == 3:
            min_conf = 0.20
        else:
            min_conf = 0.15
        
        # se detectou varias vezes aceita confiança menor
        if len(group) >= 2:
            min_conf = max(0.10, min_conf - 0.10)
        
        if best["confidence"] >= min_conf:
            final_results.append(best)

    # ordena por confianca
    final_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return final_results
