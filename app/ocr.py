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
    Configurado para máxima velocidade em CPU com foco em dígitos.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            # Usa modelo mobile (mais leve e rápido em CPU)
            # Configurações otimizadas para detecção de números em camisas
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=True,          # Detecta rotação (importante para camisas)
                lang="en",                    # Inglês para números
                use_gpu=False,                # Força uso de CPU
                show_log=False,               # Reduz logs desnecessários
                
                # Limites de tamanho - reduz processamento desnecessário
                det_limit_side_len=960,       # Limite de lado para detecção
                
                # Thresholds otimizados para números em camisas
                det_db_thresh=0.2,            # Mais sensível (pega números pequenos/foscos)
                det_db_box_thresh=0.4,        # Threshold de confiança para boxes
                det_db_unclip_ratio=2.5,      # Expande boxes (importante para números)
                
                # Configurações de reconhecimento
                rec_batch_num=1,              # Processa 1 por vez (mais rápido em CPU)
            )
            print("PaddleOCR inicializado com sucesso (modo otimizado para CPU)", flush=True)
        except Exception as e:
            print(f"Warning: Erro ao configurar PaddleOCR otimizado: {e}", flush=True)
            # Fallback para configuração básica
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            print("PaddleOCR inicializado com configuração básica", flush=True)
    return _OCR_INSTANCE

def enhance_image_for_digits(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Gera apenas 2 variantes ultra-rápidas focadas em velocidade máxima.
    Otimizado para <1 segundo de processamento total.
    """
    # Redimensiona imagem grande para acelerar processamento
    h, w = image_bgr.shape[:2]
    max_dim = 1280  # Limite para velocidade
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE rápido para contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Sharpening leve
    sharpen = cv2.addWeighted(gray_clahe, 1.3, cv2.GaussianBlur(gray_clahe, (0, 0), 1.0), -0.3, 0)

    # Apenas 2 variantes essenciais para velocidade máxima
    variants: List[Dict[str, Any]] = [
        # 1. Imagem original processada
        {"img": cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        
        # 2. Upscale moderado apenas se imagem for pequena
        {"img": cv2.resize(cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR), None, 
                          fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) if max(h, w) < 800 else cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR),
         "sx": 1.5 if max(h, w) < 800 else 1.0, 
         "sy": 1.5 if max(h, w) < 800 else 1.0},
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
    Extrai números de camisas de atletas de forma rápida e eficiente.
    Otimizado para velocidade em CPU mantendo boa precisão.
    """
    ocr = get_ocr()
    
    # Gera apenas as variantes essenciais (5 variantes vs 23+ anteriormente)
    variants = enhance_image_for_digits(image_bgr)
    
    candidates: List[Dict[str, Any]] = []

    # Processa cada variante
    for v_idx, v in enumerate(variants):
        variant = v["img"]
        sx, sy = float(v["sx"]), float(v["sy"])
        
        try:
            result = ocr.ocr(variant)
        except Exception as e:
            print(f"Erro na variante {v_idx}: {e}", flush=True)
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
        
        # Filtro de confiança mínima
        # Dígitos únicos precisam de maior confiança (evitar falsos positivos)
        min_conf = 0.60 if len(num) == 1 else 0.50
        
        # Se detectado múltiplas vezes, reduz threshold (mais confiável)
        if len(group) >= 2:
            min_conf -= 0.1
        
        if best["confidence"] >= min_conf:
            final_results.append(best)

    # Ordena por confiança (maior primeiro)
    final_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Log dos resultados
    nums_detected = ["{0}({1}%)".format(r['number'], int(r['confidence']*100)) for r in final_results]
    print(f"Números detectados: {nums_detected}", flush=True)
    return final_results
