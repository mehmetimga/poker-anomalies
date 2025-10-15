# Algorithm Quick Reference Guide

**For**: Poker Anomaly Detection Pipeline  
**Quick decision matrix for choosing the right algorithm**

---

## 🎯 Quick Decision Tree

```
Do you have labeled data?
├─ YES → Go to Supervised Learning
└─ NO  → Continue below

Do you need real-time detection (<50ms)?
├─ YES → Go to Fast Methods
└─ NO  → Continue below

Is dataset large (>10k sequences)?
├─ YES → Consider Deep Learning
└─ NO  → Use Classical Methods
```

---

## 📊 Algorithm Selection Matrix

### By Data Availability

| Scenario | Best Algorithms | Expected Accuracy | Effort |
|----------|----------------|-------------------|--------|
| **No labels, small data** | UKF + Isolation Forest | 80-85% | 2-3 days |
| **No labels, large data** | LSTM Autoencoder | 85-90% | 5-7 days |
| **With labels, small data** | XGBoost | 85-90% | 3-4 days |
| **With labels, large data** | LSTM Classifier or Transformer | 90-95% | 7-14 days |

### By Performance Requirements

| Requirement | Best Choice | Latency | Accuracy |
|-------------|-------------|---------|----------|
| **Ultra-fast (<10ms)** | ARIMA or Isolation Forest | ~5ms | 70-78% |
| **Fast (<25ms)** | UKF (current) | ~21ms | 75% |
| **Balanced** | UKF + Isolation Forest | ~25ms | 82-85% |
| **High accuracy** | XGBoost or Stacking Ensemble | ~30ms | 90-93% |
| **Research/SOTA** | Transformer | ~15ms (GPU) | 92-95% |

### By Use Case

| Use Case | Algorithm | Why |
|----------|-----------|-----|
| **Quick prototype** | Current UKF + enhancements | Already implemented, easy wins |
| **Production MVP** | UKF + Isolation Forest | No labels needed, good accuracy |
| **With labeling budget** | XGBoost | Best ROI for labeled data |
| **Research project** | LSTM or Transformer | State-of-the-art, publications |
| **Batch analysis** | Matrix Profile | Best for pattern mining |

---

## 🚀 Top 5 Recommendations

### 1. **UKF + Isolation Forest Hybrid** ⭐⭐⭐⭐⭐
- **When**: Need quick improvement without labels
- **Accuracy**: 82-85%
- **Latency**: ~25ms
- **Effort**: 2-3 days
- **Cost**: Low (CPU only)
- **Best for**: Immediate production deployment

### 2. **XGBoost Classifier** ⭐⭐⭐⭐⭐
- **When**: Have labeled data (1000+ examples)
- **Accuracy**: 88-92%
- **Latency**: ~2ms
- **Effort**: 3-4 days
- **Cost**: Low (CPU only)
- **Best for**: Highest accuracy with moderate effort

### 3. **LSTM Autoencoder** ⭐⭐⭐⭐
- **When**: Large dataset, no labels
- **Accuracy**: 85-90%
- **Latency**: ~10ms (GPU)
- **Effort**: 5-7 days
- **Cost**: Medium (GPU needed)
- **Best for**: Unsupervised with deep learning

### 4. **Stacking Ensemble** ⭐⭐⭐⭐⭐
- **When**: Need best possible accuracy
- **Accuracy**: 93-95%
- **Latency**: ~30ms
- **Effort**: 7-10 days
- **Cost**: Medium
- **Best for**: High-stakes production system

### 5. **UKF + LSTM Hybrid** ⭐⭐⭐⭐
- **When**: Want interpretability + deep learning
- **Accuracy**: 90-93%
- **Latency**: ~31ms
- **Effort**: 9-12 days
- **Cost**: Medium (GPU for training)
- **Best for**: Best of both worlds

---

## ⚡ Implementation Priority

### Week 1-2: Quick Wins
```
✅ Enhance UKF (5D state, context-aware)
✅ Add Isolation Forest
✅ Implement graph-based collusion
→ Result: +40% improvement, 85% accuracy
```

### Week 3-4: If No Labels
```
✅ Matrix Profile for pattern mining
✅ Gaussian Process for uncertainty
✅ ARIMA baseline for comparison
→ Result: Comprehensive unsupervised system
```

### Week 3-4: If Have Labels
```
✅ Label 500-1000 sequences
✅ Train XGBoost
✅ Random Forest for feature selection
→ Result: 90%+ accuracy
```

### Month 2: Advanced
```
✅ LSTM Autoencoder (unsupervised track)
✅ Stacking Ensemble (supervised track)
✅ Model serving infrastructure
→ Result: Production-ready 90%+ system
```

---

## 💡 Algorithm Cheat Sheet

### Traditional Time Series

| Algorithm | Pros | Cons | Use When |
|-----------|------|------|----------|
| **ARIMA** | Simple, fast | Linear only | Baseline comparison |
| **Holt-Winters** | Handles trends | Additive/multiplicative | Seasonal patterns |
| **Gaussian Process** | Uncertainty | Slow (O(n³)) | Small datasets |
| **Matrix Profile** | Finds motifs | Batch only | Pattern mining |

### Machine Learning

| Algorithm | Pros | Cons | Use When |
|-----------|------|------|----------|
| **Isolation Forest** | No labels, fast | No sequential | Multi-dimensional |
| **One-Class SVM** | Strong theory | Slow training | Small datasets |
| **Random Forest** | Interpretable | Needs labels | Feature importance |
| **XGBoost** | Best tabular | Needs labels | Maximum accuracy |

### Deep Learning

| Algorithm | Pros | Cons | Use When |
|-----------|------|------|----------|
| **LSTM** | Sequential | Needs many labels | Large labeled data |
| **LSTM Autoencoder** | Unsupervised | Slow training | Large unlabeled data |
| **Transformer** | SOTA | Very data-hungry | Research/huge data |
| **VAE** | Generative | Complex | Synthetic data needed |

---

## 📈 Expected Improvements

### From Current System (75% accuracy, 10% FPR)

| Change | New Accuracy | FPR | Effort |
|--------|--------------|-----|--------|
| + Multi-dimensional UKF | 78% | 9% | 1 day |
| + Multi-threshold detection | 80% | 8% | 1 day |
| + Isolation Forest | 85% | 6% | 2 days |
| + Graph collusion | 85% | 5% (collusion) | 3 days |
| + XGBoost (with labels) | 90% | 3% | 4 days |
| + Stacking Ensemble | 93% | 2% | 7 days |
| + LSTM Autoencoder | 88% | 4% | 7 days |

---

## 🎓 Learning Curve

### Easy (1-3 days)
- ARIMA/SARIMA
- Exponential Smoothing
- Isolation Forest
- UKF enhancements

### Medium (3-7 days)
- Gaussian Process
- Matrix Profile
- One-Class SVM
- Random Forest
- XGBoost
- LSTM Autoencoder

### Hard (7-14 days)
- LSTM Classifier
- Transformer
- VAE
- Stacking Ensemble
- GNN-based methods

---

## 💰 Cost Analysis

### Low Cost (<$100/month)
- All CPU-based methods
- UKF + Isolation Forest
- XGBoost
- ARIMA
- Random Forest

### Medium Cost ($100-500/month)
- LSTM Autoencoder (GPU training)
- LSTM Classifier
- Stacking Ensemble

### High Cost (>$500/month)
- Transformer
- Large-scale GNN
- Continuous retraining with GPUs

---

## 🔥 Hot Takes

### Most Underrated
**Matrix Profile**: Incredible for finding patterns, everyone sleeps on it

### Most Overrated
**Transformer**: Needs massive data, overkill for poker

### Best Bang for Buck
**UKF + Isolation Forest**: 85% accuracy, 3 days, no labels

### Hidden Gem
**Gaussian Process**: Amazing uncertainty quantification

### Production Workhorse
**XGBoost**: Just works, fast, accurate

### Research Star
**Graph Neural Networks**: Future of multi-player detection

---

## 🎯 My Recommendation (Opinionated)

### Phase 1 (Week 1-2): Foundation
```python
# Implement these in order:
1. Multi-dimensional UKF (1 day)
2. Isolation Forest integration (1 day)
3. Graph-based collusion (2 days)
4. Multi-threshold detection (1 day)

# Result: 85% accuracy, production-ready
# Investment: 1 week
```

### Phase 2 (Week 3-6): If Have Labels
```python
# Label data while building:
1. Label 500 sequences (2 weeks, can parallelize)
2. Implement XGBoost (2 days)
3. Feature engineering (2 days)
4. Hyperparameter tuning (1 week)

# Result: 90% accuracy
# Investment: 4 weeks
```

### Phase 2 (Week 3-6): If No Labels
```python
# Deep learning alternative:
1. LSTM Autoencoder (4 days implementation)
2. Training infrastructure (2 days)
3. Hyperparameter search (1 week)
4. Threshold calibration (2 days)

# Result: 87% accuracy
# Investment: 3 weeks
```

### Phase 3 (Month 2-3): Excellence
```python
# For the gold standard:
1. Combine all methods into stacking ensemble
2. Add online learning
3. Build monitoring dashboard
4. A/B test in production

# Result: 93%+ accuracy, bulletproof system
```

---

## 📚 Further Reading

### Beginners
- Start with ARIMA/Exponential Smoothing
- Move to Isolation Forest
- Then UKF enhancements

### Intermediate
- XGBoost for supervised learning
- Matrix Profile for pattern mining
- Gaussian Process for uncertainty

### Advanced
- LSTM Autoencoder for deep unsupervised
- Stacking Ensemble for maximum accuracy
- GNNs for network analysis

---

## ❓ FAQ

**Q: Which is fastest?**  
A: ARIMA (~5ms) or Isolation Forest (~1ms)

**Q: Which is most accurate?**  
A: Stacking Ensemble (93%) or Transformer (95% with huge data)

**Q: Which needs least data?**  
A: UKF (works with 1 player), Gaussian Process (10+ samples)

**Q: Which is most interpretable?**  
A: UKF (state-space model) or ARIMA (coefficients)

**Q: Best without labels?**  
A: UKF + Isolation Forest (82-85%)

**Q: Best with labels?**  
A: XGBoost (90%) or Stacking (93%)

**Q: Easiest to implement?**  
A: Isolation Forest or ARIMA (2-3 days)

**Q: Best for production?**  
A: UKF + Isolation Forest + XGBoost (when labels available)

---

## 🎬 TL;DR

**Need quick results?** → UKF enhancements + Isolation Forest (3 days, 85%)

**Have labels?** → XGBoost (4 days, 90%)

**Have time & GPU?** → LSTM Autoencoder (1 week, 87%)

**Want best accuracy?** → Stacking Ensemble (2 weeks, 93%)

**Research project?** → Transformer or GNN (3 weeks, 95%)

**My choice?** → **UKF + Isolation Forest → Add XGBoost when labels ready**

---

**Last Updated**: October 2025  
**For detailed implementation**: See `IMPROVEMENTS_AND_ALTERNATIVES.md`

