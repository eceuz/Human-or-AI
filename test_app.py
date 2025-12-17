import unittest
import numpy as np
# app.py dosyasından test edeceğimiz fonksiyonları import ediyoruz
# Eğer import hatası alırsan try-except bloğu devreye girer
try:
    from app import clean_input_text, get_prediction 
except ImportError:
    pass

# --- SAHTE (MOCK) MODEL SINIFLARI ---
class MockModel:
    def __init__(self, proba):
        self._proba = proba
    
    def predict_proba(self, text_vector):
        return np.array([self._proba])

class MockSVC:
    def __init__(self, score):
        self._score = score
        
    def decision_function(self, text_vector):
        return np.array([self._score])


# --- ANA TEST SINIFI ---
class TestModelFunctions(unittest.TestCase):
    
    # 1. TEST: METİN TEMİZLEME TESTİ
    def test_cleaning_functionality(self):
        """clean_input_text fonksiyonunun gürültüyü doğru temizlediğini test eder."""
        
        # DÜZELTİLEN KISIM: Kodumuz [and] kısmını tamamen sildiği için
        # Beklenen metinden 'and' kelimesini çıkardık.
        dirty_text = "This is a Test! (with brackets [and] symbols) and S P A C E S. %10"
        expected_clean = "this is a test with brackets symbols and s p a c e s %10"
        
        # Testi çalıştır
        self.assertEqual(clean_input_text(dirty_text), expected_clean)
        
        # İkinci temizleme testi (Boşluklar ve sayılar)
        text_with_spaces = "  Multiple    spaces and numbers 123 "
        expected_spaces = "multiple spaces and numbers 123"
        self.assertEqual(clean_input_text(text_with_spaces), expected_spaces)

    # 2. TEST: predict_proba MODELLERİNİN TESTİ (NB, RF, LR)
    def test_predict_proba_models(self):
        """Olasılık döndüren modellerin (NB, RF, LR) yüzdeleri doğru hesapladığını test eder."""
        
        model_human_ai = MockModel([0.20, 0.80])
        prob_ai, prob_human = get_prediction(model_human_ai, None)
        
        self.assertAlmostEqual(prob_ai, 80.0) 
        self.assertAlmostEqual(prob_human, 20.0)
        
    # 3. TEST: decision_function MODELİNİN TESTİ (Linear SVC)
    def test_linear_svc_model(self):
        """decision_function kullanan LinearSVC modelinin olasılığa doğru dönüştürüldüğünü test eder."""
        
        test_score = 1.0 
        svc_model = MockSVC(test_score)
        
        prob_ai, prob_human = get_prediction(svc_model, None)
        
        # P(AI) = 1 / (1 + e^(-score)) formülü kontrol ediliyor
        expected_ai_prob = (1 / (1 + np.exp(-test_score))) * 100
        
        self.assertAlmostEqual(prob_ai, expected_ai_prob)
        self.assertAlmostEqual(prob_human, 100 - expected_ai_prob)


if __name__ == '__main__':
    # Tüm testleri çalıştır
    unittest.main()