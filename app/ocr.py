from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR

# Otimizações para CPU - 1 thread é mais rápido para inferência
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

# Instância global do OCR para evitar custo de carregamento repetido
_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_ocr() -> PaddleOCR:
    """
    Inicializa e retorna instância singleton do PaddleOCR.
    Configurado para detectar TODOS os números em cenas com múltiplos atletas.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            # Configuração otimizada para múltiplas detecções
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=False,          # OFF para velocidade
                lang="en",                    # Números
                use_gpu=False,                # CPU
                show_log=False,               # Sem logs
                
                # Thresholds muito sensíveis para capturar TODOS os números
                det_limit_side_len=1600,      # Aumentado para preservar detalhes
                det_db_thresh=0.15,           # MUITO sensível
                det_db_box_thresh=0.3,        # Baixo para capturar tudo
                det_db_unclip_ratio=2.5,      # Expande bem os boxes
                
                # Reconhecimento
                rec_batch_num=6,              # Batch processing
                drop_score=0.2,               # Aceita confiança baixa
                max_text_length=3,            # Números de camisa: 1-3 dígitos
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
    """
    Pré-processamento AVANÇADO para detectar números de camisas.
    Técnicas específicas para atletas em movimento, baixa resolução e condições variadas.
    """
    # Preserva tamanho adequado
    h, w = image_bgr.shape[:2]
    max_dim = 1600
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w
    elif max(h, w) < 800:  # Upscale se muito pequena
        scale = 1200 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w
    
    # Converte para LAB para melhor processamento de luminância
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ========== TÉCNICA 1: CLAHE Agressivo + Denoising ==========
    # Reduz ruído primeiro
    l_denoised = cv2.fastNlMeansDenoising(l, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # CLAHE muito agressivo para contraste máximo
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_denoised)
    
    # Reconstrói imagem LAB
    lab_enhanced = cv2.merge([l_clahe, a, b])
    enhanced1 = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Sharpening agressivo usando unsharp mask
    gaussian = cv2.GaussianBlur(enhanced1, (0, 0), 2.0)
    enhanced1 = cv2.addWeighted(enhanced1, 2.0, gaussian, -1.0, 0)
    
    # ========== TÉCNICA 2: Morphological Enhancement ==========
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Equalização de histograma
    gray_eq = cv2.equalizeHist(gray)
    
    # Morphological gradient para realçar bordas dos números
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray_eq, cv2.MORPH_GRADIENT, kernel)
    
    # Combina com original
    enhanced2 = cv2.addWeighted(gray_eq, 0.7, morph, 0.3, 0)
    
    # Sharpening
    enhanced2 = cv2.addWeighted(enhanced2, 1.5, cv2.GaussianBlur(enhanced2, (0, 0), 1.0), -0.5, 0)
    enhanced2_bgr = cv2.cvtColor(enhanced2, cv2.COLOR_GRAY2BGR)
    
    # ========== TÉCNICA 3: Adaptive Threshold Múltiplo ==========
    # CLAHE na escala de cinza
    gray_clahe = clahe.apply(gray)
    
    # Bilateral filter para preservar bordas e remover ruído
    bilateral = cv2.bilateralFilter(gray_clahe, 9, 75, 75)
    
    # Adaptive threshold otimizado para números
    adaptive1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    
    # Segunda variação com parâmetros diferentes
    adaptive2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 15, 3)
    
    # Combina os dois adaptive thresholds
    adaptive_combined = cv2.bitwise_or(adaptive1, adaptive2)
    
    # Morfologia para limpar ruído
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    adaptive_clean = cv2.morphologyEx(adaptive_combined, cv2.MORPH_CLOSE, kernel_small)
    adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_OPEN, kernel_small)
    
    adaptive_bgr = cv2.cvtColor(adaptive_clean, cv2.COLOR_GRAY2BGR)
    
    # ========== TÉCNICA 4: Contrast Stretching ==========
    # Normaliza histograma para máximo contraste
    norm = cv2.normalize(gray_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    # Aplica threshold OTSU
    _, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverte se fundo for escuro
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    
    otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    
    # Retorna 4 melhores variantes
    variants = [
        # 1. CLAHE + Denoise + Sharpen (melhor para contraste)
        {"img": enhanced1, "sx": 1.0, "sy": 1.0},
        
        # 2. Morphological enhancement (bom para bordas)
        {"img": enhanced2_bgr, "sx": 1.0, "sy": 1.0},
        
        # 3. Adaptive threshold combinado (ótimo para fundos variados)
        {"img": adaptive_bgr, "sx": 1.0, "sy": 1.0},
        
        # 4. OTSU (backup para casos difíceis)
        {"img": otsu_bgr, "sx": 1.0, "sy": 1.0},
    ]
    
    return variants


def _only_digit_sequences(text: str) -> List[str]:
    """Extrai sequências de dígitos e dígitos individuais do texto."""
    # Remove espaços e caracteres não-numéricos para focar nos dígitos
    text_clean = re.sub(r'[^\d]', '', text)
    if not text_clean:
        return []
    
    # Extrai sequências contíguas de 1 a 4 dígitos
    seqs = re.findall(r"\d{1,4}", text)
    cleaned: List[str] = []
    for s in seqs:
        if s.isdigit():
            cleaned.append(str(int(s)))  # Remove zeros à esquerda
    
    # Se não encontrou sequências, tenta pegar dígitos individuais
    if not cleaned and text_clean:
        for char in text_clean:
            if char.isdigit():
                cleaned.append(char)
    
    return list(set(cleaned))  # Remove duplicatas


def extract_jersey_numbers(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extrai números de camisas ULTRA-RÁPIDO.
    Otimizado para <1 segundo em CPU.
    """
    import time
    start = time.time()
    
    ocr = get_ocr()
    
    # Apenas 1 variante otimizada
    variants = enhance_image_for_digits(image_bgr)
    prep_time = time.time() - start
    
    candidates: List[Dict[str, Any]] = []

    # Processa variante única
    ocr_start = time.time()
    for v in variants:
        variant = v["img"]
        sx, sy = float(v["sx"]), float(v["sy"])
        
        try:
            result = ocr.ocr(variant)
        except Exception as e:
            print(f"Erro OCR: {e}", flush=True)
            continue
            
        if not result:
            continue
        
        # Processa cada detecção
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
                
                # Extrai sequências de dígitos - MELHORADO
                digit_seqs = _only_digit_sequences(text)
                
                # Se não encontrou, tenta extrair TODOS os dígitos do texto
                if not digit_seqs:
                    text_digits_only = re.sub(r'[^\d]', '', text)
                    if text_digits_only:
                        # Remove zeros à esquerda
                        text_digits_only = text_digits_only.lstrip('0') or '0'
                        if 1 <= len(text_digits_only) <= 3:  # 1-3 dígitos
                            digit_seqs = [text_digits_only]
                
                # Filtra números impossíveis (muito grandes)
                digit_seqs = [s for s in digit_seqs if s and 1 <= len(s) <= 2 and int(s) <= 99]
                
                if not digit_seqs or box is None:
                    continue
                
                # Reescala bbox para o espaço original
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
                
                # Adiciona cada sequência de dígitos encontrada
                for seq in digit_seqs:
                    if 1 <= len(seq) <= 2:  # Números de camisa: 1-2 dígitos (0-99)
                        # Ajusta confiança baseado no match
                        seq_conf = conf if len(seq) == len(text.strip()) else conf * 0.80
                        candidates.append({
                            "number": seq,
                            "confidence": round(seq_conf, 4),
                            "bbox": bbox,
                        })

    if not candidates:
        return []

    from collections import defaultdict
    
    # Agrupa por número
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[c["number"]].append(c)

    # Filtra e retorna apenas as melhores detecções
    final_results: List[Dict[str, Any]] = []
    for num, group in grouped.items():
        # Valida número (0-99 é range típico de camisas esportivas)
        try:
            num_int = int(num)
            if num_int < 0 or num_int > 99:  # Apenas 0-99
                continue
        except ValueError:
            continue
        
        # Pega a detecção com maior confiança
        best = max(group, key=lambda x: x["confidence"])
        
        # Filtro de confiança MUITO permissivo para não perder números
        min_conf = 0.30 if len(num) == 1 else 0.25  # Muito baixo
        
        # Se detectado múltiplas vezes, aceita ainda menor
        if len(group) >= 2:
            min_conf = 0.20  # Quase tudo
        
        if best["confidence"] >= min_conf:
            final_results.append(best)

    # Ordena por confiança (maior primeiro)
    final_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Log com timing
    total_time = time.time() - start
    ocr_time = time.time() - ocr_start
    nums_detected = ["{0}({1}%)".format(r['number'], int(r['confidence']*100)) for r in final_results]
    print(f"Detectado: {nums_detected} | Prep: {prep_time*1000:.0f}ms | OCR: {ocr_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms", flush=True)
    return final_results
