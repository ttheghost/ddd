/**
 * @file transformer.cpp
 * @brief Single-Head Attention Transformer Seq2Seq English-to-French Translator.
 *
 * Architecture
 *
 *   Encoder : English Embedding + Positional Encoding → Single-Head Self-Attention → Feed-Forward → Context Matrix H_enc
 *   Decoder : French Embedding + Positional Encoding → Single-Head Causal Self-Attention → Single-Head Cross-Attention (with H_enc) → Feed-Forward → Dense Classifier → Softmax
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

// Helper: Stack Token Embeddings into Sequence Matrix (d_model, seq_len)

static dl::Tensor<float> stack_embeddings(const std::vector<dl::Tensor<float>> &embeds) {
  const size_t d_model = embeds[0].shape()[0];
  const size_t seq_len = embeds.size();
  dl::Tensor<float> seq({d_model, seq_len});
  float *s_ptr = seq.data();

  for (size_t col = 0; col < seq_len; ++col) {
    const float *e_ptr = embeds[col].data();
    for (size_t r = 0; r < d_model; ++r) {
      s_ptr[r * seq_len + col] = e_ptr[r];
    }
  }
  return seq;
}

// Main Seq2Seq Transformer Translator

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
    constexpr size_t d_model = 32;
    dl::Adam<float> adam(0.005f);

    dl::Embedding<float> enc_embed(en_vocab.size(), d_model, adam);
    dl::PositionalEncoding<float> enc_pos(d_model);
    dl::SingleHeadAttention<float> enc_attn(d_model, adam, /*causal_mask=*/false);
    dl::Dense<float> enc_ff(d_model, d_model, adam);

    dl::Embedding<float> dec_embed(fr_vocab.size(), d_model, adam);
    dl::PositionalEncoding<float> dec_pos(d_model);
    dl::SingleHeadAttention<float> dec_self_attn(d_model, adam, /*causal_mask=*/true);
    dl::SingleHeadAttention<float> dec_cross_attn(d_model, adam, /*causal_mask=*/false);
    dl::Dense<float> dec_ff(d_model, d_model, adam);
    dl::Dense<float> dec_out(d_model, fr_vocab.size(), adam);
    dl::Softmax<float> softmax;

    // Training Loop
    constexpr size_t epochs = 120;
    std::cout << "Training Single-Head Transformer Translator (" << epochs << " epochs)...\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      float total_loss = 0.f;

      for (const auto &[en_text, fr_text] : corpus) {
        auto en_tokens = tokenize(en_text);
        auto fr_tokens = tokenize(fr_text);

        std::vector<size_t> target_ids;
        for (const auto &w : fr_tokens) target_ids.push_back(fr_vocab.get_id(w));
        target_ids.push_back(EOS_ID);

        // Encoder Forward
        std::vector<size_t> en_ids;
        for (const auto &w : en_tokens) en_ids.push_back(en_vocab.get_id(w));

        std::vector<dl::Tensor<float>> enc_embeds;
        for (size_t id : en_ids) enc_embeds.push_back(enc_embed.forward(id));

        dl::Tensor<float> X_enc = enc_pos.forward(stack_embeddings(enc_embeds));
        dl::Tensor<float> H_attn = enc_attn.forward(X_enc, X_enc, X_enc);
        dl::Tensor<float> H_enc = enc_ff.forward(H_attn);

        // Decoder Input Sequence starting with <SOS>
        std::vector<size_t> dec_input_ids = {SOS_ID};
        for (size_t i = 0; i < target_ids.size() - 1; ++i) {
          dec_input_ids.push_back(target_ids[i]);
        }

        std::vector<dl::Tensor<float>> dec_embeds;
        for (size_t id : dec_input_ids) dec_embeds.push_back(dec_embed.forward(id));

        dl::Tensor<float> X_dec = dec_pos.forward(stack_embeddings(dec_embeds));
        dl::Tensor<float> Y_self = dec_self_attn.forward(X_dec, X_dec, X_dec);
        dl::Tensor<float> Y_cross = dec_cross_attn.forward(Y_self, H_enc, H_enc);
        dl::Tensor<float> Y_ff = dec_ff.forward(Y_cross);

        // Word Predictions & Losses per target step
        const size_t seq_len = target_ids.size();
        dl::Tensor<float> grad_dec_out_seq({fr_vocab.size(), seq_len});
        grad_dec_out_seq.fill(0.f);

        for (size_t step = 0; step < seq_len; ++step) {
          size_t target_id = target_ids[step];

          // Extract column slice at current step
          dl::Tensor<float> h_step = Y_ff.slice(1, step, step + 1);
          auto logits = dec_out.forward(h_step);
          auto pred = softmax.forward(logits);

          total_loss += cross_entropy(pred, target_id);

          dl::Tensor<float> grad_logits = pred;
          grad_logits.data()[target_id] -= 1.0f;

          auto grad_h = dec_out.backward(grad_logits);
          const float *gh_ptr = grad_h.data();
          float *gseq_ptr = grad_dec_out_seq.data();

          for (size_t r = 0; r < d_model; ++r) {
            gseq_ptr[r * seq_len + step] = gh_ptr[r];
          }
        }

        // Backward Pass through Decoder & Encoder
        dl::Tensor<float> grad_Y_cross = dec_ff.backward(grad_dec_out_seq);

        dl::Tensor<float> grad_Y_self({d_model, seq_len}), grad_H_k({d_model, en_ids.size()}), grad_H_v({d_model, en_ids.size()});
        dec_cross_attn.backward(grad_Y_cross, grad_Y_self, grad_H_k, grad_H_v);

        dl::Tensor<float> grad_H_enc = grad_H_k + grad_H_v;
        dl::Tensor<float> grad_H_attn = enc_ff.backward(grad_H_enc);

        dl::Tensor<float> grad_X_enc_q({d_model, en_ids.size()}), grad_X_enc_k({d_model, en_ids.size()}), grad_X_enc_v({d_model, en_ids.size()});
        enc_attn.backward(grad_H_attn, grad_X_enc_q, grad_X_enc_k, grad_X_enc_v);

        dl::Tensor<float> grad_X_enc_total = enc_pos.backward(grad_X_enc_q + grad_X_enc_k + grad_X_enc_v);
        for (size_t i = 0; i < en_ids.size(); ++i) {
          enc_embed.backward(grad_X_enc_total.slice(1, i, i + 1));
        }

        dl::Tensor<float> grad_X_dec_q({d_model, seq_len}), grad_X_dec_k({d_model, seq_len}), grad_X_dec_v({d_model, seq_len});
        dec_self_attn.backward(grad_Y_self, grad_X_dec_q, grad_X_dec_k, grad_X_dec_v);

        dl::Tensor<float> grad_X_dec_total = dec_pos.backward(grad_X_dec_q + grad_X_dec_k + grad_X_dec_v);
        for (size_t i = 0; i < dec_input_ids.size(); ++i) {
          dec_embed.backward(grad_X_dec_total.slice(1, i, i + 1));
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
    std::cout << "TRANSFORMER TRANSLATION DEMO (English -> French)\n";
    std::cout << std::string(60, '=') << "\n";

    auto translate = [&](const std::string &en_text) {
      auto en_tokens = tokenize(en_text);
      std::vector<size_t> en_ids;
      for (const auto &w : en_tokens) en_ids.push_back(en_vocab.get_id(w));

      std::vector<dl::Tensor<float>> enc_embeds;
      for (size_t id : en_ids) enc_embeds.push_back(enc_embed.forward(id));

      dl::Tensor<float> X_enc = enc_pos.forward(stack_embeddings(enc_embeds));
      dl::Tensor<float> H_attn = enc_attn.forward(X_enc, X_enc, X_enc);
      dl::Tensor<float> H_enc = enc_ff.forward(H_attn);

      std::vector<size_t> dec_input_ids = {SOS_ID};
      std::string translated;

      for (size_t step = 0; step < 10; ++step) {
        std::vector<dl::Tensor<float>> dec_embeds;
        for (size_t id : dec_input_ids) dec_embeds.push_back(dec_embed.forward(id));

        dl::Tensor<float> X_dec = dec_pos.forward(stack_embeddings(dec_embeds));
        dl::Tensor<float> Y_self = dec_self_attn.forward(X_dec, X_dec, X_dec);
        dl::Tensor<float> Y_cross = dec_cross_attn.forward(Y_self, H_enc, H_enc);
        dl::Tensor<float> Y_ff = dec_ff.forward(Y_cross);

        dl::Tensor<float> h_step = Y_ff.slice(1, Y_ff.shape()[1] - 1, Y_ff.shape()[1]);
        auto logits = dec_out.forward(h_step);
        auto pred = softmax.forward(logits);
        size_t pred_id = pred.argmax();

        if (pred_id == EOS_ID) break;

        std::string word = fr_vocab.get_word(pred_id);
        if (word != "<PAD>" && word != "<SOS>") {
          if (!translated.empty()) translated += " ";
          translated += word;
        }

        dec_input_ids.push_back(pred_id);
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
