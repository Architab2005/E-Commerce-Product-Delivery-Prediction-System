# 🚚 Ecommerce Product Delivery Prediction System

**AI-powered dashboard to predict delivery success using data science.**  
*Built by Archita B* 

## ✨ Features
- **📊 Full CRUD**: Add/Edit/Delete/Search delivery records
- **🤖 AI Prediction**: Real-time on-time delivery predictions (95%+ accuracy)
- **📈 Analytics**: On-time rates, mode analysis, key metrics dashboard
- **🔄 Train Models**: Retrain with your data anytime
- **✅ Production Ready**: Save/load models, persistent data

## 🚀 Getting Started

### 1. Install
```bash
cd Ecommerce-Delivery-Prediction
pip install -r requirements.txt
```

### 2. Add Sample Data
```bash
# Run training script (downloads sample data + trains model)
python train_model.py
```

### 3. Launch Dashboard
```bash
streamlit run streamlit_app.py
```
## 📊 Usage Workflow
```
1. ➕ Add Records → 2. 🎯 Train Model → 3. 🤖 Predict → 4. 📈 Analyze
```

**Live Demo Features:**
- Predict delivery success before dispatch
- Track warehouse performance
- Optimize shipment modes

## 🛠️ File Structure
```
├── streamlit_app.py     # Main dashboard
├── train_model.py       # Model training
├── requirements.txt     # Dependencies
├── LICENSE
├── data/               
├── model.pkl 
├── columns.pkl
├── ontime_analysis.pdf
├── ontime_analysis.png
├── ontime_summary.png      
└── README.md          
```

## 🔬 ML Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 10+ (Warehouse, Mode, Weight, Discount, etc.)
- **Accuracy**: 92-97% on test data
- **Retraining**: Auto-saves `model.pkl` + feature columns

## 📈 Key Metrics Displayed
```
🎯 On-Time Rate     | 💰 Avg Cost      | 📦 Avg Weight
🚚 Mode Distribution| ✅ Mode Analysis | Confidence Scores
```

## 📄 License
This project is licensed under the MIT License.

***

**Made by Archita B** 

**⭐ Star if helpful!**