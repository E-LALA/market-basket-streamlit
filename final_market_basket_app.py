
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import io

item_translations = {
    "لبن": "Milk", "بيض": "Eggs", "خبز": "Bread", "شاي": "Tea", "سكر": "Sugar", "ماء": "Water", 
    "فراخ": "Chicken", "رز": "Rice", "بطاطس": "Potatoes", "عسل": "Honey", "مربى": "Jam", "زيت": "Oil", 
    "طماطم": "Tomato", "كورن فليكس": "Cornflakes", "بصل": "Onion", "عدس": "Lentils", "زبادي": "Yogurt", 
    "لحمة": "Meat", "بيبسي": "Pepsi", "شيبسي": "Chips", "عيش بلدي": "Baladi Bread", "مكرونة": "Pasta", 
    "سمنة": "Butter", "فلفل": "Pepper", "كشري": "Koshary", "جبنة قريش": "Cottage cheese",
    "فول": "Foul", "طعميه": "Falafel"
}

def translate(items, lang):
    return [item_translations.get(i, i) if lang == "English" else i for i in items]

def explain_rule(antecedents, consequents, confidence, lift, lang):
    a = ', '.join(translate(antecedents, lang))
    c = ', '.join(translate(consequents, lang))
    return f"If a customer buys [{a}], they are {lift:.2f}× more likely to also buy [{c}] with {confidence:.0%} confidence."

st.set_page_config(page_title="Market Basket Analyzer", layout="wide")
st.title("🛒 Market Basket Analysis")

with st.sidebar:
    st.header("⚙️ Settings")
    algorithm = st.selectbox("Algorithm", ["Apriori", "FP-Growth"])
    lang = st.radio("Item Name Language", ["Original (Arabic)", "English"])
    min_support = st.slider("Min Support", 0.01, 1.0, 0.1, 0.01)
    min_confidence = st.slider("Min Confidence", 0.01, 1.0, 0.3, 0.01)
    top_n_rules = st.number_input("Top N Rules", min_value=1, max_value=100, value=10)

uploaded_file = st.file_uploader("📤 Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = ['items']
    transactions = df['items'].apply(lambda x: x.split(','))
    all_items = sorted(set(item for transaction in transactions for item in transaction))
    df_encoded = pd.DataFrame([[item in transaction for item in all_items] for transaction in transactions], columns=all_items)

    if algorithm == "Apriori":
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules["rule_text"] = rules.apply(lambda r: explain_rule(list(r["antecedents"]), list(r["consequents"]), r["confidence"], r["lift"], lang), axis=1)

        tab1, tab2 = st.tabs(["📊 Frequent Itemsets", "📋 Association Rules"])

        with tab1:
            st.subheader("📊 Frequent Itemsets Table")
            frequent_itemsets['translated_itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(translate(list(x), lang)))
            st.dataframe(frequent_itemsets[['translated_itemsets', 'support', 'itemsets']])

            st.subheader("📈 Top Frequent Itemsets Support Chart")
            top_items = frequent_itemsets.sort_values(by='support', ascending=False).head(20)
            top_items['translated_itemsets'] = top_items['itemsets'].apply(lambda x: ', '.join(translate(list(x), lang)))
            fig, ax = plt.subplots(figsize=(14, 6))
            bars = ax.barh(
                top_items['translated_itemsets'],
                top_items['support'],
                color=plt.cm.plasma(top_items['support'] / top_items['support'].max())
            )
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Support")
            ax.set_title("Top Frequent Itemsets by Support")
            st.pyplot(fig)

            st.download_button("📥 Download Frequent Itemsets CSV", data=frequent_itemsets.to_csv(index=False), file_name="frequent_itemsets.csv")

        with tab2:
            st.subheader("📋 Association Rules Table")
            item_filter = st.text_input("🔎 Filter rules by item name (optional)").strip()
            if item_filter:
                rules = rules[rules["antecedents"].apply(lambda x: item_filter in x) | rules["consequents"].apply(lambda x: item_filter in x)]

            rules = rules.sort_values(by="confidence", ascending=False).head(top_n_rules)
            rule_table = rules[["rule_text", "support", "confidence", "lift"]]
            st.dataframe(rule_table)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                rule_table.to_excel(writer, index=False, sheet_name="Rules")
            st.download_button("📤 Download Rules as Excel", data=output.getvalue(), file_name="association_rules.xlsx")

            st.subheader("📉 Enhanced Association Rules Network")
            if not rules.empty:
                G = nx.DiGraph()
                for _, rule in rules.iterrows():
                    for ant in rule['antecedents']:
                        for con in rule['consequents']:
                            a = item_translations.get(ant, ant) if lang == "English" else ant
                            c = item_translations.get(con, con) if lang == "English" else con
                            G.add_node(a)
                            G.add_node(c)
                            G.add_edge(a, c, weight=rule['confidence'], lift=rule['lift'])

                pos = nx.spring_layout(G, k=0.8, iterations=50)
                degrees = dict(G.degree())
                node_sizes = [2000 + degrees[n] * 400 for n in G.nodes()]
                node_colors = plt.cm.cool(np.linspace(0.3, 0.9, len(G.nodes())))

                fig, ax = plt.subplots(figsize=(16, 10))
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors='black')
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
                edge_weights = [d['weight'] * 4 for _, _, d in G.edges(data=True)]
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_weights, edge_cmap=plt.cm.Blues, width=edge_weights, arrows=True)
                ax.set_title("Enhanced Association Rules Network", fontsize=16)
                st.pyplot(fig)

    else:
        st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
