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
    Otimizado para múltiplos números mantendo velocidade.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            # Configuração balanceada: velocidade + múltiplas detecções
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=False,          # OFF para velocidade
                lang="en",                    # Números
                use_gpu=False,                # CPU
                show_log=False,               # Sem logs
                
                # Thresholds ajustados para múltiplos números
                det_limit_side_len=1280,      # Limite maior
                det_db_thresh=0.2,            # Mais sensível para pegar todos
                det_db_box_thresh=0.4,        # Reduzido para capturar mais
                det_db_unclip_ratio=2.2,      # Expande boxes
                
                # Otimizações
                rec_batch_num=6,              # Batch processing
                drop_score=0.3,               # Aceita confiança menor
                max_text_length=4,            # Máximo 3 dígitos + margem
            )
            print("PaddleOCR inicializado (otimizado para múltiplos números)", flush=True)
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
    2 variantes rápidas otimizadas para múltiplos números.
    Balanço ideal entre velocidade e detecção de todos os números.
    """
    # Redimensiona para tamanho ideal
    h, w = image_bgr.shape[:2]
    max_dim = 1280  # Aumentado para capturar mais detalhes
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    
    # Pré-processamento otimizado
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE forte para contraste
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpening para realçar números
    enhanced = cv2.addWeighted(enhanced, 1.4, cv2.GaussianBlur(enhanced, (3, 3), 0), -0.4, 0)
    
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # 2 variantes para capturar todos os números
    variants = [
        # 1. Imagem processada normal
        {"img": enhanced_bgr, "sx": 1.0, "sy": 1.0},
        
        # 2. Upscale para números pequenos/distantes
        {"img": cv2.resize(enhanced_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
         "sx": 1.5, "sy": 1.5},
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
                
                # Extrai sequências de dígitos
                digit_seqs = _only_digit_sequences(text)
                
                # Se não encontrou, tenta extrair apenas dígitos do texto
                if not digit_seqs:
                    text_digits_only = re.sub(r'[^\d]', '', text)
                    if text_digits_only and 1 <= len(text_digits_only) <= 3:  # Máximo 3 dígitos (0-999)
                        digit_seqs = [text_digits_only]
                
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
                    if 1 <= len(seq) <= 3:  # Números de camisa geralmente tem 1-3 dígitos
                        # Ajusta confiança baseado no match
                        seq_conf = conf if len(seq) == len(text.strip()) else conf * 0.85
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
        # Valida número (1-999 é range típico de camisas)
        try:
            num_int = int(num)
            if num_int < 0 or num_int > 999:
                continue
        except ValueError:
            continue
        
        # Pega a detecção com maior confiança
        best = max(group, key=lambda x: x["confidence"])
        
        # Filtro de confiança mais permissivo para capturar todos os números
        min_conf = 0.45 if len(num) == 1 else 0.35  # Reduzido para pegar mais
        
        # Se detectado múltiplas vezes, aceita confiança ainda menor
        if len(group) >= 2:
            min_conf -= 0.15
        
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
