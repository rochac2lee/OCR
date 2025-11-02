from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

# Instância global do OCR para evitar custo de carregamento repetido
_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_ocr() -> PaddleOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            # Inicialização para usar modelos VL.
            # O UVDoc ou o PP-OCRv5 são modelos avançados (server)
            # que carregam a inteligência visual-linguística.
            _OCR_INSTANCE = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                # O modelo de detecção padrão (PP-OCRv5_server_det)
                # O modelo de reconhecimento VL
                rec_model_name='en_PP-OCRv5_server_rec', # Nome comum para o modelo server/VL
                # Ou rec_model_name='UVDoc', dependendo da sua versão do PaddleX/PaddleOCR
                
                # Parâmetros de otimização
                det_db_thresh=0.3, 
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=2.0,
                det_limit_side_len=960,
                # use_cuda=False # Deixado fora para evitar o erro 'Unknown argument'
            )
        except Exception as e:
            # ... (código de fallback)
            print(f"Warning: Erro ao configurar PaddleOCR com parâmetros otimizados: {e}", flush=True)
            # Fallback para a configuração básica se houver erro
            try:
                _OCR_INSTANCE = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    # Tenta forçar o modelo Server/VL para o fallback também
                    rec_model_name='en_PP-OCRv5_server_rec', 
                )
            except Exception:
                # Último fallback
                _OCR_INSTANCE = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                )
    return _OCR_INSTANCE

def enhance_image_for_digits(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Retorna apenas as variantes mais eficazes para balancear velocidade e precisão."""
    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE para aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Filtro bilateral para reduzir ruído preservando bordas
    denoised = cv2.bilateralFilter(gray_clahe, d=7, sigmaColor=75, sigmaSpace=75)

    # Unsharp masking para nitidez
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpen = cv2.addWeighted(denoised, 1.8, gaussian, -0.8, 0)

    # Adaptive threshold para capturar números em diferentes condições de contraste
    adap = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    
    # Adaptive threshold invertido (para números claros em fundo escuro)
    adap_inv = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    
    # OTSU threshold para binarização adaptativa
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inversão de imagem para números claros em fundo escuro
    inverted = cv2.bitwise_not(gray_clahe)
    inverted_sharpen = cv2.addWeighted(inverted, 1.8, cv2.GaussianBlur(inverted, (0, 0), 2.0), -0.8, 0)
    
    # Adaptive threshold na imagem invertida (para reforçar números claros)
    adap_inverted = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    # Variantes base: original, sharpen, adaptive threshold, adaptive invertido, OTSU
    # E variantes invertidas para números claros
    variants: List[Dict[str, Any]] = [
        {"img": img, "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(adap_inv, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(inverted_sharpen, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
        {"img": cv2.cvtColor(adap_inverted, cv2.COLOR_GRAY2BGR), "sx": 1.0, "sy": 1.0},
    ]
    
    # Upscales mais agressivos para capturar números pequenos
    # Inclui sharpen (normal e invertido) e adaptive threshold
    base_sharpen = variants[1]["img"]
    base_adap = variants[2]["img"]
    base_adap_inv = variants[3]["img"]  # Adaptive invertido
    base_inv_sharpen = variants[5]["img"]  # Invertido sharpen
    
    # Usa upscales maiores para melhorar detecção de números pequenos
    for base, base_name in [
        (base_sharpen, "sharpen"), 
        (base_adap, "adap"), 
        (base_adap_inv, "adap_inv"),
        (base_inv_sharpen, "inv_sharpen")
    ]:
        for scale in (1.5, 2.0, 2.5, 3.0):  # Adiciona 3.0x para números muito pequenos
            up = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            variants.append({"img": up, "sx": scale, "sy": scale})

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


def _refine_with_crop(ocr: PaddleOCR, full_bgr: np.ndarray, bbox: Dict[str, int]) -> Optional[Dict[str, Any]]:
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    h_full, w_full = full_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    if x >= w_full or y >= h_full:
        return None
    crop = full_bgr[y:min(y + h, h_full), x:min(x + w, w_full)].copy()
    if crop.size == 0:
        return None
    # Aumenta e binariza o recorte para forçar leitura dos dígitos
    crop_up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    crop_proc = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    try:
        r = ocr.ocr(crop_proc)
    except Exception:
        return None
    if not r:
        return None
    best_seq: Optional[str] = None
    best_conf = 0.0
    for line in r:
        if not line:
            continue
        for det in line:
            if not det or len(det) < 2 or det[1] is None or len(det[1]) < 2:
                continue
            text = det[1][0] if isinstance(det[1][0], str) else ""
            try:
                conf = float(det[1][1])
            except Exception:
                conf = 0.0
            for seq in _only_digit_sequences(text):
                if 1 <= len(seq) <= 4 and conf > best_conf:
                    best_seq, best_conf = seq, conf
    if best_seq is None:
        return None
    return {"number": best_seq, "confidence": float(best_conf)}


def extract_jersey_numbers(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    ocr = get_ocr()

    # Além das variantes globais, também processa regiões específicas da imagem
    # onde geralmente ficam os números dos atletas (parte superior central)
    h, w = image_bgr.shape[:2]
    
    # Define regiões de interesse (ROI) - geralmente números ficam na parte superior do torso
    rois = []
    # ROI 1: Parte superior esquerda
    rois.append({"x": 0, "y": 0, "w": w//2, "h": h//2})
    # ROI 2: Parte superior direita
    rois.append({"x": w//2, "y": 0, "w": w//2, "h": h//2})
    # ROI 3: Parte central superior
    rois.append({"x": w//4, "y": 0, "w": w//2, "h": h//2})
    # ROI 4: Imagem completa (já será processada pelas variantes)
    
    variants = enhance_image_for_digits(image_bgr)
    # Adiciona crops de regiões de interesse
    for roi in rois[:3]:  # Pega apenas as 3 primeiras ROIs
        roi_crop = image_bgr[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
        if roi_crop.size > 0:
            roi_variants = enhance_image_for_digits(roi_crop)
            # Ajusta as variantes para incluir offset da ROI
            for v in roi_variants:
                v["roi_offset"] = {"x": roi["x"], "y": roi["y"]}
            variants.extend(roi_variants)
    
    candidates: List[Dict[str, Any]] = []

    # Coleta TODOS os candidatos sem filtros iniciais - filtragem será feita depois
    all_texts_detected = []  # Para debug
    for v_idx, v in enumerate(variants):
        variant = v["img"]
        sx, sy = float(v["sx"]), float(v["sy"])
        try:
            result = ocr.ocr(variant)
        except Exception as e:
            print(f"ERROR na variante {v_idx}: {e}", flush=True)
            continue
        if not result:
            continue
        
        # Debug: coleta todos os textos detectados
        for line in result:
            if line:
                for det in line or []:
                    if det and len(det) >= 2 and det[1]:
                        text = det[1][0] if isinstance(det[1][0], str) else str(det[1][0])
                        all_texts_detected.append(text)
                        
                        # TAMBÉM processa caracteres individuais do texto para capturar dígitos
                        # que podem estar misturados com letras ou não detectados separadamente
                        for char in text:
                            if char.isdigit() and char not in ['0']:  # Ignora zero que geralmente é falso positivo
                                all_texts_detected.append(char)
        
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
                
                # Processa o texto para extrair números - tenta múltiplas estratégias
                digit_seqs = _only_digit_sequences(text)
                
                # Estratégia 1: Se não encontrou sequências, tenta pegar qualquer sequência de dígitos
                if not digit_seqs:
                    text_digits_only = re.sub(r'[^\d]', '', text)
                    if text_digits_only and len(text_digits_only) <= 4:
                        digit_seqs = [text_digits_only]
                
                # Estratégia 2: Extrai dígitos individuais mesmo que estejam misturados com letras
                if not digit_seqs and text:
                    # Remove espaços e zeros à esquerda
                    cleaned = re.sub(r'[\s]', '', text)
                    cleaned = re.sub(r'^0+', '', cleaned) if cleaned else ''
                    if cleaned.isdigit() and 1 <= len(cleaned) <= 4:
                        digit_seqs = [cleaned]
                
                # Estratégia 3: Tenta extrair dígitos individuais do texto
                if not digit_seqs:
                    individual_digits = [c for c in text if c.isdigit() and c != '0']
                    if individual_digits:
                        digit_seqs = individual_digits
                
                # Se ainda não encontrou nada numérico válido, pula
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
                
                # Aplica offset da ROI se existir
                roi_offset = v.get("roi_offset", {"x": 0, "y": 0})
                bbox = {
                    "x": x_min + roi_offset["x"],
                    "y": y_min + roi_offset["y"],
                    "w": w,
                    "h": h,
                }
                # Aceita TODOS os sequências de 1-4 dígitos encontradas
                # Usa a confiança do OCR mesmo para dígitos extraídos de sequências maiores
                for seq in digit_seqs:
                    if 1 <= len(seq) <= 4:
                        # Para sequências extraídas, mantém a confiança original
                        # Se for dígito individual de texto maior, usa confiança um pouco reduzida
                        seq_conf = conf if len(seq) == len(text.strip()) else conf * 0.9
                        candidates.append(
                            {
                                "number": seq,
                                "confidence": round(seq_conf, 4),
                                "bbox": bbox,
                            }
                        )

    if not candidates:
        return []

    from collections import defaultdict
    
    # DEBUG: Print resumo do que foi detectado
    print(f"\n=== DEBUG OCR ===", flush=True)
    print(f"Textos detectados pelo PaddleOCR (amostra): {all_texts_detected[:20]}", flush=True)
    unique_nums_seen = set()
    for c in candidates:
        unique_nums_seen.add(c["number"])
    print(f"Total candidatos coletados: {len(candidates)}, números únicos: {sorted(unique_nums_seen)}", flush=True)
    print(f"==================\n", flush=True)

    # Agrupa por número (sem filtragem prévia - aceita tudo)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[c["number"]].append(c)

    # Para cada número único, mantém apenas a detecção mais confiável
    # Isso remove duplicatas do mesmo número detectado múltiplas vezes
    final_results: List[Dict[str, Any]] = []
    for num, group in grouped.items():
        # Valida se o número faz sentido (1-4 dígitos, número válido)
        try:
            num_int = int(num)
            if num_int < 1 or num_int > 9999:  # Números muito altos são provavelmente falsos positivos
                continue
        except ValueError:
            continue
        
        # Pega a detecção com maior confiança
        best = max(group, key=lambda x: x["confidence"])
        
        # Filtra por confiança mínima: dígitos únicos precisam de confiança maior
        min_conf = 0.65 if len(num) == 1 else 0.55  # Números de 1 dígito precisam ser mais confiáveis
        
        # Se apareceu múltiplas vezes, aceita confiança um pouco menor
        if len(group) >= 2:
            min_conf -= 0.1
        
        if best["confidence"] >= min_conf:
            final_results.append(best)

    # Ordena por confiança (maior primeiro)
    final_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    print(f"DEBUG: Resultados finais (após filtragem): {[r['number'] for r in final_results]}", flush=True)
    return final_results
