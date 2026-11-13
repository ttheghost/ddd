/**
 * @file lstm.cpp
 * @brief LSTM Sequence-to-Sequence (Seq2Seq) English-to-French Translator.
 *
 * Architecture
 *
 *   Encoder : English Embedding → 4-gate LSTMCell → Context States (h_enc, c_enc)
 *   Decoder : French Embedding + 4-gate LSTMCell (initialized with h_enc, c_enc) → Dense Classifier → Softmax
 *
 * Vocabulary
 *
 *   Special tokens: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
 */

#include <dl/dl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

// Special Token Constants

constexpr size_t PAD_ID = 0;
constexpr size_t SOS_ID = 1;
constexpr size_t EOS_ID = 2;
constexpr size_t UNK_ID = 3;

// Tokenizer & Vocabulary Builder

struct Vocabulary {
  std::map<std::string, size_t> word2id;
  std::map<size_t, std::string> id2word;

  Vocabulary() {
    add_word("<PAD>"); // 0
    add_word("<SOS>"); // 1
    add_word("<EOS>"); // 2
    add_word("<UNK>"); // 3
  }

  void add_word(const std::string &word) {
    if (word2id.find(word) == word2id.end()) {
      size_t id = word2id.size();
      word2id[word] = id;
      id2word[id] = word;
    }
  }

  [[nodiscard]] size_t get_id(const std::string &word) const {
    auto it = word2id.find(word);
    return (it != word2id.end()) ? it->second : UNK_ID;
  }

  [[nodiscard]] std::string get_word(size_t id) const {
    auto it = id2word.find(id);
    return (it != id2word.end()) ? it->second : "<UNK>";
  }

  [[nodiscard]] size_t size() const { return word2id.size(); }
};

static std::vector<std::string> tokenize(const std::string &text) {
  std::vector<std::string> tokens;
  std::string clean;
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c)) || std::isspace(static_cast<unsigned char>(c))) {
      clean += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  std::stringstream ss(clean);
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

// Cross-Entropy Loss

static float cross_entropy(const dl::Tensor<float> &pred, size_t target_id) {
  return -std::log(std::max(pred.data()[target_id], 1e-7f));
}

int main() {
  try {
    // Parallel English-French Corpus
    std::vector<std::pair<std::string, std::string>> corpus = {
      {"hello", "bonjour"},
      {"good morning", "bonjour"},
      {"good night", "bonne nuit"},
      {"thank you", "merci"},
      {"goodbye", "au revoir"},
      {"welcome", "bienvenue"},
      {"yes", "oui"},
      {"no", "non"},
      {"see you soon", "a bientot"},
      {"have a good day", "bonne journee"},
      {"how are you", "comment vas tu"},
      {"i love machine learning", "j aime l apprentissage automatique"}
    };

    Vocabulary en_vocab, fr_vocab;
    for (const auto &[en, fr] : corpus) {
      for (const auto &w : tokenize(en)) en_vocab.add_word(w);
      for (const auto &w : tokenize(fr)) fr_vocab.add_word(w);
    }

    std::cout << "English Vocabulary Size: " << en_vocab.size() << "\n";
    std::cout << "French Vocabulary Size : " << fr_vocab.size() << "\n\n";

    // Model Parameters & Layers
    constexpr size_t embed_dim = 32;
    constexpr size_t hidden_dim = 64;

    dl::Adam<float> adam(0.005f);

    dl::Embedding<float> enc_embed(en_vocab.size(), embed_dim, adam);
    dl::LSTMCell<float> enc_cell(embed_dim, hidden_dim, adam);

    dl::Embedding<float> dec_embed(fr_vocab.size(), embed_dim, adam);
    dl::LSTMCell<float> dec_cell(embed_dim, hidden_dim, adam);
    dl::Dense<float> dec_out(hidden_dim, fr_vocab.size(), adam);
    dl::Softmax<float> softmax;

    // Training Loop
    constexpr size_t epochs = 120;
    std::cout << "Training Seq2Seq LSTM Translator (" << epochs << " epochs)...\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      float total_loss = 0.f;

      for (const auto &[en_text, fr_text] : corpus) {
        auto en_tokens = tokenize(en_text);
        auto fr_tokens = tokenize(fr_text);

        // Build target sequence ending with <EOS>
        std::vector<size_t> target_ids;
        for (const auto &w : fr_tokens) target_ids.push_back(fr_vocab.get_id(w));
        target_ids.push_back(EOS_ID);

        // Encoder Forward
        dl::Tensor<float> h_enc({hidden_dim, 1}), c_enc({hidden_dim, 1});
        h_enc.fill(0.f);
        c_enc.fill(0.f);

        for (const auto &w : en_tokens) {
          size_t id = en_vocab.get_id(w);
          auto x = enc_embed.forward(id);
          enc_cell.forward(x, h_enc, c_enc, h_enc, c_enc);
        }

        // Decoder Forward & Backward (Teacher Forcing)
        dl::Tensor<float> h_dec = h_enc;
        dl::Tensor<float> c_dec = c_enc;
        size_t input_token = SOS_ID;

        for (size_t target_id : target_ids) {
          auto x = dec_embed.forward(input_token);
          dl::Tensor<float> h_next({hidden_dim, 1}), c_next({hidden_dim, 1});
          dec_cell.forward(x, h_dec, c_dec, h_next, c_next);

          auto logits = dec_out.forward(h_next);
          auto pred = softmax.forward(logits);

          total_loss += cross_entropy(pred, target_id);

          // Combined Softmax+CE Gradient: pred - one_hot(target_id)
          dl::Tensor<float> grad_logits = pred;
          grad_logits.data()[target_id] -= 1.0f;

          // Backprop through Decoder output -> LSTMCell -> Embedding
          auto grad_h = dec_out.backward(grad_logits);
          dl::Tensor<float> grad_c({hidden_dim, 1});
          grad_c.fill(0.f);

          dl::Tensor<float> grad_x({embed_dim, 1}), grad_h_prev({hidden_dim, 1}), grad_c_prev({hidden_dim, 1});
          dec_cell.backward(grad_h, grad_c, grad_x, grad_h_prev, grad_c_prev);
          dec_embed.backward(grad_x);

          h_dec = h_next;
          c_dec = c_next;
          input_token = target_id;
        }
      }

      if ((epoch + 1) % 20 == 0 || epoch == epochs - 1) {
        std::cout << "Epoch " << std::setw(3) << (epoch + 1)
                  << "/" << epochs << "  Average Loss = "
                  << std::fixed << std::setprecision(4)
                  << (total_loss / static_cast<float>(corpus.size())) << "\n";
      }
    }

    // Translation Evaluation Demo
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "LSTM TRANSLATION DEMO (English -> French)\n";
    std::cout << std::string(60, '=') << "\n";

    auto translate = [&](const std::string &en_text) {
      auto en_tokens = tokenize(en_text);
      dl::Tensor<float> h_enc({hidden_dim, 1}), c_enc({hidden_dim, 1});
      h_enc.fill(0.f);
      c_enc.fill(0.f);

      for (const auto &w : en_tokens) {
        size_t id = en_vocab.get_id(w);
        auto x = enc_embed.forward(id);
        enc_cell.forward(x, h_enc, c_enc, h_enc, c_enc);
      }

      std::string translated;
      dl::Tensor<float> h_dec = h_enc;
      dl::Tensor<float> c_dec = c_enc;
      size_t input_token = SOS_ID;

      for (size_t step = 0; step < 10; ++step) {
        auto x = dec_embed.forward(input_token);
        dl::Tensor<float> h_next({hidden_dim, 1}), c_next({hidden_dim, 1});
        dec_cell.forward(x, h_dec, c_dec, h_next, c_next);

        auto logits = dec_out.forward(h_next);
        auto pred = softmax.forward(logits);
        size_t pred_id = pred.argmax();

        if (pred_id == EOS_ID) break;

        std::string word = fr_vocab.get_word(pred_id);
        if (word != "<PAD>" && word != "<SOS>") {
          if (!translated.empty()) translated += " ";
          translated += word;
        }

        h_dec = h_next;
        c_dec = c_next;
        input_token = pred_id;
      }
      return translated;
    };

    std::vector<std::string> test_phrases = {
      "hello",
      "good morning",
      "thank you",
      "goodbye",
      "good night",
      "welcome",
      "see you soon",
      "how are you",
      "i love machine learning"
    };

    for (const auto &phrase : test_phrases) {
      std::cout << "English: \"" << phrase << "\"\n";
      std::cout << "French : \"" << translate(phrase) << "\"\n";
      std::cout << std::string(40, '-') << "\n";
    }

    std::cout << "\nDone.\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
