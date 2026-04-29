import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================
# CONFIG & PAGE SETUP
# ==========================
st.set_page_config(
    page_title="🚚 Delivery Prediction System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_file_paths():
    return {
        'data': "data/E_Commerce.csv",
        'model': "model.pkl", 
        'columns': "columns.pkl"
    }

paths = get_file_paths()
os.makedirs("data", exist_ok=True)      

# ==========================
# DATA FUNCTIONS with ID SUPPORT
# ==========================
@st.cache_data
def load_data():
    if not os.path.exists(paths['data']):
        columns = ["ID"] + [
            "Warehouse_block", "Mode_of_Shipment", "Customer_care_calls",
            "Customer_rating", "Cost_of_the_Product", "Prior_purchases",
            "Product_importance", "Gender", "Discount_offered", 
            "Weight_in_gms", "Reached.on.Time_Y.N"
        ]
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(paths['data'])
    if 'ID' not in df.columns:
        df.insert(0, 'ID', range(1, len(df) + 1))
    elif df['ID'].isna().any() or df['ID'].duplicated().any():
        df['ID'] = range(1, len(df) + 1)
    return df

def save_data(df):
    df.to_csv(paths['data'], index=False)
    st.cache_data.clear()
    st.success("✅ Data saved!")

# ==========================
# MODEL FUNCTIONS
# ==========================
@st.cache_resource
def load_model():
    try:
        if os.path.exists(paths['model']) and os.path.exists(paths['columns']):
            with open(paths['model'], 'rb') as f:
                model = pickle.load(f)
            with open(paths['columns'], 'rb') as f:
                columns = pickle.load(f)
            return model, columns
    except:
        pass
    return None, None

def preprocess_data(df):
    model, columns = load_model()
    if model is None:
        return None
    
    df_no_id = df.drop(columns=['ID'], errors='ignore')
    df_processed = pd.get_dummies(df_no_id, drop_first=True)
    missing_cols = set(columns) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0
    df_processed = df_processed.reindex(columns=columns, fill_value=0)
    return df_processed

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("🚚 Delivery Dashboard")
page = st.sidebar.selectbox(
    "Select Action",
    ["📊 View Data", "➕ Add Data", "✏️ Edit/Delete", "🔍 Search", 
     "🤖 Predict", "🎯 Train Model", "📈 Stats"]
)

# ==========================
# PAGE 1: VIEW DATAS
# ==========================
if page == "📊 View Data":
    st.header("📊 All Delivery Records")
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("On Time", len(df[df['Reached.on.Time_Y.N'] == 1]))
    col3.metric("Delayed", len(df[df['Reached.on.Time_Y.N'] == 0]))
    
    st.dataframe(df, use_container_width=True)

# ==========================
# PAGE 2: ADD DATA (MANUAL ID INPUT)
# ==========================
elif page == "➕ Add Data":
    st.header("➕ Add New Delivery Record")
    
    with st.form("add_form", clear_on_submit=True):
        st.markdown("---")
        
        # ID & Basic Info
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.markdown("**🆔 Record ID**")
            new_id = st.number_input("Enter ID", min_value=1, max_value=999999, step=1)
        with col2:
            st.markdown("**📦 Shipment Details**")
            warehouse = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'E', 'F'])
            mode = st.selectbox("Mode of Shipment", ['Flight', 'Ship', 'Road', 'Regular Air', 'Express Air'])
        with col3:
            st.markdown("**📞 Customer Info**")
            calls = st.slider("Customer Care Calls", 0, 15, 2)
            rating = st.slider("Customer Rating (1-5)", 1, 5, 2)
        
        # Product Details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💰 Product Details**")
            cost = st.number_input("Cost of Product (₹)", 50.0, 50000.0, 250.0)
            discount = st.number_input("Discount Offered (%)", 0.0, 80.0, 12.5)
        with col2:
            st.markdown("**📏 Product Specs**")
            weight = st.number_input("Weight (grams)", 100.0, 100000.0, 1800.0)
            prior = st.slider("Prior Purchases", 0, 20, 1)
        
        # Additional Details
        col1, col2, col3 = st.columns(3)
        importance = col1.selectbox("Product Importance", ['low', 'medium', 'high', 'critical'])
        gender = col2.selectbox("Gender", ['M', 'F'])
        reached = col3.selectbox("Reached on Time?", ['Yes', 'No'])
        
        submitted = st.form_submit_button("🚀 ADD RECORD", type="primary", use_container_width=True)
        
        if submitted:
            df = load_data()
            if new_id in df['ID'].values:
                st.error(f"❌ **ID {new_id} already exists!** Please choose a unique ID.")
                st.stop()
            
            new_record = {
                'ID': new_id,
                'Warehouse_block': warehouse,
                'Mode_of_Shipment': mode,
                'Customer_care_calls': calls,
                'Customer_rating': rating,
                'Cost_of_the_Product': cost,
                'Prior_purchases': prior,
                'Product_importance': importance,
                'Gender': gender,
                'Discount_offered': discount,
                'Weight_in_gms': weight,
                'Reached.on.Time_Y.N': 1 if reached == 'Yes' else 0
            }
            
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df = df.sort_values('ID').reset_index(drop=True)
            save_data(df)
            
            st.success(f"""
            🎉 **Record Added Successfully!**
            🆔 **ID**: {new_id}
            🚚 **Mode**: {mode}
            💰 **Cost**: ₹{cost:,.0f}
            ✅ **Status**: {'On Time' if reached == 'Yes' else 'Delayed'}
            """)
            st.balloons()

# ==========================
# PAGE 3: EDIT/DELETE
# ==========================
elif page == "✏️ Edit/Delete":
    st.header("✏️ Manage Records")
    df = load_data()
    
    if df.empty:
        st.warning("👆 Add some records first!")
    else:
        st.subheader("📋 All Records")
        for i, row in df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1])
                
                col1.markdown(f"""
                **🆔 ID {row['ID']}** | 
                {row['Mode_of_Shipment']} | 
                ₹{row['Cost_of_the_Product']:.0f} | 
                {row['Weight_in_gms']/1000:.1f}kg | 
                {'✅' if row['Reached.on.Time_Y.N'] else '❌'}
                """)
                
                if col2.button("✏️ Edit", key=f"edit_{row['ID']}"):
                    st.session_state.edit_id = row['ID']
                    st.rerun()
                
                if col3.button("🗑️ Delete", key=f"del_{row['ID']}"):
                    df = df[df['ID'] != row['ID']].reset_index(drop=True)
                    df['ID'] = range(1, len(df) + 1)
                    save_data(df)
                    st.success(f"✅ ID {row['ID']} deleted!")
                    st.rerun()
        
        if 'edit_id' in st.session_state:
            edit_id = st.session_state.edit_id
            row = df[df['ID'] == edit_id].iloc[0]
            
            st.subheader(f"✏️ Edit Record **🆔 ID {edit_id}**")
            
            with st.form(f"edit_{edit_id}"):
                col1, col2 = st.columns(2)
                with col1:
                    mode = st.selectbox("Mode", df['Mode_of_Shipment'].unique(), 
                                      index=list(df['Mode_of_Shipment']).index(row['Mode_of_Shipment']))
                    discount = st.number_input("Discount (%)", value=row['Discount_offered'])
                with col2:
                    weight = st.number_input("Weight (g)", value=row['Weight_in_gms'])
                    status = st.selectbox("Status", ["Yes", "No"], 
                                        index=0 if row['Reached.on.Time_Y.N'] else 1)
                
                if st.form_submit_button("💾 Update Record"):
                    df.loc[df['ID'] == edit_id, 'Mode_of_Shipment'] = mode
                    df.loc[df['ID'] == edit_id, 'Discount_offered'] = discount
                    df.loc[df['ID'] == edit_id, 'Weight_in_gms'] = weight
                    df.loc[df['ID'] == edit_id, 'Reached.on.Time_Y.N'] = 1 if status == "Yes" else 0
                    save_data(df)
                    st.success(f"✅ ID {edit_id} updated successfully!")
                    del st.session_state.edit_id
                    st.rerun()

# ==========================
# PAGE 4: SEARCH
# ==========================
elif page == "🔍 Search":
    st.header("🔍 Search & Filter Records")
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    id_filter = col1.text_input("🆔 ID", placeholder="e.g., 1, 5")
    mode_filter = col2.multiselect("🚚 Mode", df['Mode_of_Shipment'].unique())
    warehouse_filter = col3.multiselect("🏢 Warehouse", df['Warehouse_block'].unique())
    status_filter = col4.selectbox("✅ Status", ['All', 'On Time', 'Delayed'])
    
    filtered = df.copy()
    if id_filter:
        filtered = filtered[filtered['ID'].astype(str).str.contains(id_filter, na=False)]
    if mode_filter:
        filtered = filtered[filtered['Mode_of_Shipment'].isin(mode_filter)]
    if warehouse_filter:
        filtered = filtered[filtered['Warehouse_block'].isin(warehouse_filter)]
    if status_filter == 'On Time':
        filtered = filtered[filtered['Reached.on.Time_Y.N'] == 1]
    elif status_filter == 'Delayed':
        filtered = filtered[filtered['Reached.on.Time_Y.N'] == 0]
    
    st.dataframe(filtered, use_container_width=True)
    st.caption(f"🆔 Showing {len(filtered)} of {len(df)} total records")

# ==========================
# PAGE 5: PREDICT
# ==========================
elif page == "🤖 Predict":
    st.header("🤖 AI Delivery Prediction")
    model, columns = load_model()
    
    if model is None:
        st.error("❌ No trained model found! Go to **Train Model** tab first.")
        st.info("💡 Add 10+ records → Train Model → Predict here!")
    else:
        st.success("✅ Model loaded! Ready to predict.")
        
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                warehouse = st.selectbox("Warehouse", ['A', 'B', 'C', 'D', 'E', 'F'])
                mode = st.selectbox("Mode", ['Flight', 'Ship', 'Road', 'Regular Air', 'Express Air'])
                calls = st.slider("Customer Calls", 0, 15, 2)
            with col2:
                cost = st.number_input("Cost (₹)", 50.0, 50000.0, 250.0)
                weight = st.number_input("Weight (g)", 100.0, 100000.0, 1800.0)
                discount = st.slider("Discount %", 0.0, 80.0, 12.5)
            
            col1, col2 = st.columns(2)
            with col1:
                rating = st.slider("Rating", 1, 5, 2)
                prior = st.slider("Prior Purchases", 0, 20, 1)
            with col2:
                importance = st.selectbox("Importance", ['low', 'medium', 'high', 'critical'])
                gender = st.selectbox("Gender", ['M', 'F'])
            
            predict_btn = st.form_submit_button("🔮 MAKE PREDICTION", type="primary", use_container_width=True)
            
            if predict_btn:
                input_df = pd.DataFrame([{
                    'Warehouse_block': warehouse,
                    'Mode_of_Shipment': mode,
                    'Customer_care_calls': calls,
                    'Customer_rating': rating,
                    'Cost_of_the_Product': cost,
                    'Prior_purchases': prior,
                    'Product_importance': importance,
                    'Gender': gender,
                    'Discount_offered': discount,
                    'Weight_in_gms': weight
                }])
                
                X_pred = preprocess_data(input_df)
                pred = model.predict(X_pred)[0]
                probs = model.predict_proba(X_pred)[0]
                
                col1, col2 = st.columns([2, 1])
                status = "✅ ON TIME" if pred == 1 else "❌ DELAYED"
                confidence = max(probs) * 100
                
                col1.metric("Prediction", status, delta=None)
                col2.metric("Confidence", f"{confidence:.1f}%")
                
                st.balloons()

# ==========================
# PAGE 6: TRAIN MODEL
# ==========================
elif page == "🎯 Train Model":
    st.header("🎯 Train Machine Learning Model")
    
    df = load_data()
    col1, col2 = st.columns(2)
    col1.metric("Records Available", len(df))
    col2.metric("Model Status", "✅ Ready" if os.path.exists("model.pkl") else "❌ Train Needed")
    
    if st.button("🚀 TRAIN NEW MODEL", type="primary", use_container_width=True):
        with st.spinner("Training AI model... Please wait"):
            if len(df) < 5:
                st.error("❌ Need **5+ records** to train! Add data first.")
            else:
                df_no_id = df.drop('ID', axis=1, errors='ignore')
                df_processed = pd.get_dummies(df_no_id, drop_first=True)
                X = df_processed.drop('Reached.on.Time_Y.N', axis=1)
                y = df_processed['Reached.on.Time_Y.N']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                accuracy = accuracy_score(y_test, model.predict(X_test))
                
                with open(paths['model'], 'wb') as f:
                    pickle.dump(model, f)
                with open(paths['columns'], 'wb') as f:
                    pickle.dump(X.columns, f)
                
                st.success(f"""
                🎉 **MODEL TRAINED SUCCESSFULLY!**
                📊 **Accuracy**: {accuracy:.2%}
                💾 **Files saved**: model.pkl, columns.pkl
                """)
                st.balloons()
                st.rerun()

# ==========================
# PAGE 7: STATS
# ==========================
elif page == "📈 Stats":
    st.header("📈 Delivery Analytics Dashboard")
    df = load_data()
    
    if df.empty:
        st.info("👆 Add some records to see analytics!")
    else:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        on_time_rate = len(df[df['Reached.on.Time_Y.N'] == 1]) / len(df) * 100
        avg_cost = df['Cost_of_the_Product'].mean()
        avg_weight = df['Weight_in_gms'].mean()
        avg_discount = df['Discount_offered'].mean()
        
        col1.metric("🎯 On-Time Rate", f"{on_time_rate:.1f}%")
        col2.metric("💰 Avg Cost", f"₹{avg_cost:.0f}")
        col3.metric("📦 Avg Weight", f"{avg_weight/1000:.1f}kg")
        col4.metric("🔥 Avg Discount", f"{avg_discount:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚚 Mode Distribution")
            mode_counts = df['Mode_of_Shipment'].value_counts()
            st.bar_chart(mode_counts)

        with col2:
           st.subheader("✅ On-Time Rate by Mode")
           if len(df) > 0:
               mode_ontime = df.groupby('Mode_of_Shipment')['Reached.on.Time_Y.N'].mean() * 100
               chart_data = pd.DataFrame({'Mode': mode_ontime.index,'On_Time_Rate': mode_ontime.values})
               st.bar_chart(chart_data.set_index('Mode'))
           else:
               st.info("👆 Add records first!")

# Footer
st.markdown("---")
st.markdown("""**Developed by Archita B | B.Tech CSE'26 | Tech‑Driven Business Solutions Enthusiast **""")
