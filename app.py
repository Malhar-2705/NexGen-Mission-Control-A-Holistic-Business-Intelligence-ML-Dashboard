import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="NexGen Mission Control", layout="wide")


@st.cache_data
def load_data():
    try:
        orders = pd.read_csv("orders.csv")
        delivery = pd.read_csv("delivery_performance.csv")
        routes = pd.read_csv("routes_distance.csv")
        costs = pd.read_csv("cost_breakdown.csv")
        fleet = pd.read_csv("vehicle_fleet.csv")
        warehouses = pd.read_csv("warehouse_inventory.csv")
        feedback = pd.read_csv("customer_feedback.csv")
    except FileNotFoundError as e:
        st.error(f"Error: Missing file. Make sure all 7 CSVs are in the same folder as app.py. Details: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return None, None, None, None

    try:
        data = pd.merge(delivery, orders, on="Order_ID", how="left")
        data = pd.merge(data, routes, on="Order_ID", how="left")
        data = pd.merge(data, costs, on="Order_ID", how="left")
    except KeyError as e:
        st.error(f"A merge error occurred. A key column is missing: {e}")
        return None, None, None, None

    data['Total_Cost_INR'] = data['Fuel_Cost'] + data['Labor_Cost'] + data['Vehicle_Maintenance'] + \
                             data['Insurance'] + data['Packaging_Cost'] + data['Technology_Platform_Fee'] + \
                             data['Other_Overhead']
    data['Cost_Per_KM'] = data['Total_Cost_INR'] / data['Distance_KM'].replace(0, pd.NA)

    return data, fleet, warehouses, feedback


@st.cache_resource
def build_cluster_models(data, feedback):
    route_features = ['Distance_KM', 'Total_Cost_INR', 'Traffic_Delay_Minutes']
    route_data = data[route_features]

    route_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    processed_route_data = route_preprocessor.fit_transform(route_data)

    route_model = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    route_model.fit(processed_route_data)

    customer_data_merged = pd.merge(feedback, data[['Order_ID', 'Order_Value_INR']], on='Order_ID', how='left')

    customer_features = ['Rating', 'Order_Value_INR']
    customer_data = customer_data_merged[customer_features]

    customer_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    processed_customer_data = customer_preprocessor.fit_transform(customer_data)

    customer_model = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    customer_model.fit(processed_customer_data)

    return route_model, route_preprocessor, customer_model, customer_preprocessor


st.title("NexGen Logistics: Mission Control Dashboard 🚀")

data, fleet, warehouses, feedback = load_data()

if data is None:
    st.stop()

try:
    route_model, route_preprocessor, customer_model, customer_preprocessor = build_cluster_models(data, feedback)
except Exception as e:
    st.error(f"Error building ML cluster models: {e}")
    st.stop()

st.sidebar.header("Dashboard Filters")
selected_priority = st.sidebar.multiselect(
    "Filter by Priority", options=data["Priority"].unique(), default=data["Priority"].unique()
)
selected_carrier = st.sidebar.multiselect(
    "Filter by Carrier", options=data["Carrier"].unique(), default=data["Carrier"].unique()
)
selected_origin = st.sidebar.multiselect(
    "Filter by Origin", options=data["Origin"].unique(), default=data["Origin"].unique()
)

filtered_data = data[
    data["Priority"].isin(selected_priority) &
    data["Carrier"].isin(selected_carrier) &
    data["Origin"].isin(selected_origin)
    ]

if filtered_data.empty:
    st.warning("No data matches your filter criteria.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance & Cost",
    "🌍 Sustainability Analysis",
    "🗣️ Customer Feedback",
    "🧠 ML Cluster Analysis",
    "📦 Warehouse & Inventory"
])

with tab1:
    st.header("Performance & Cost Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", f"{len(filtered_data)}")
    col2.metric("Total Cost (INR)", f"₹{filtered_data['Total_Cost_INR'].sum():,.0f}")
    col3.metric("Avg. Customer Rating", f"{filtered_data['Customer_Rating'].mean():.2f} ★")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("On-Time vs. Delayed by Carrier")
        carrier_performance = filtered_data.groupby(['Carrier', 'Delivery_Status']).size().reset_index(name='Count')
        fig1 = px.bar(
            carrier_performance, x="Carrier", y="Count", color="Delivery_Status",
            title="Carrier Delivery Status",
            color_discrete_map={'On-Time': 'green', 'Slightly-Delayed': 'orange', 'Severely-Delayed': 'red'}
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Total Cost Breakdown")
        cost_cols = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost',
                     'Technology_Platform_Fee']
        cost_totals = filtered_data[cost_cols].sum().reset_index(name='Total_Cost_INR')
        cost_totals = cost_totals.rename(columns={'index': 'Cost_Category'})
        fig2 = px.pie(cost_totals, names="Cost_Category", values="Total_Cost_INR", title="Cost Component Breakdown")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Cost per KM by Route")
        fig3 = px.scatter(
            filtered_data, x="Distance_KM", y="Cost_Per_KM", color="Route",
            hover_data=["Order_ID"], title="Cost per KM vs. Distance by Route"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Customer Rating Distribution by Carrier")
        fig4 = px.box(
            filtered_data, x="Carrier", y="Customer_Rating", color="Carrier",
            title="Customer Rating Distribution", points="all"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Filtered Data Table")
    with st.expander("Click to view/hide filtered data"):
        st.dataframe(filtered_data)


    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')


    csv_data = convert_df_to_csv(filtered_data)
    st.download_button(
        label="📥 Download Filtered Data as CSV", data=csv_data,
        file_name="filtered_logistics_data.csv", mime="text/csv"
    )

with tab2:
    st.header("Fleet Sustainability Analysis")
    st.write("This dashboard analyzes the sustainability of the vehicle fleet.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avg. CO2 Emissions by Vehicle Type")
        avg_co2 = fleet.groupby('Vehicle_Type')['CO2_Emissions_Kg_per_KM'].mean().reset_index()
        fig5 = px.bar(
            avg_co2.sort_values('CO2_Emissions_Kg_per_KM'),
            x="Vehicle_Type", y="CO2_Emissions_Kg_per_KM",
            color="Vehicle_Type", title="Avg. CO2 Emissions (Kg/km)"
        )
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        st.subheader("Fuel Efficiency vs. Vehicle Age")
        fig6 = px.scatter(
            fleet, x="Age_Years", y="Fuel_Efficiency_KM_per_L",
            color="Vehicle_Type", title="Fuel Efficiency Degrades with Age",
            hover_data=["Vehicle_ID"]
        )
        st.plotly_chart(fig6, use_container_width=True)
    st.markdown("---")
    st.subheader("Vehicle Fleet Data")
    st.dataframe(fleet)

with tab3:
    st.header("Customer Feedback Analysis")
    st.write("This tab analyzes customer feedback for the orders selected by the filters.")
    feedback_merged = pd.merge(feedback, filtered_data, on="Order_ID", how="inner")

    if feedback_merged.empty:
        st.warning("No customer feedback found for the orders matching your filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Issue Categories")
            issue_counts = feedback_merged['Issue_Category'].value_counts().reset_index()
            fig7 = px.pie(
                issue_counts, names="Issue_Category", values="count", title="Reported Issue Breakdown"
            )
            st.plotly_chart(fig7, use_container_width=True)
        with col2:
            st.subheader("Avg. Rating by Carrier")
            carrier_ratings = feedback_merged.groupby('Carrier')['Rating'].mean().reset_index()
            fig8 = px.bar(
                carrier_ratings.sort_values('Rating'),
                x="Carrier", y="Rating", title="Average Customer Rating (1-5)"
            )
            fig8.update_layout(yaxis_range=[1, 5])
            st.plotly_chart(fig8, use_container_width=True)

        st.markdown("---")
        st.subheader("Actionable Feedback (Low Ratings: 1-2 ★)")
        poor_reviews = feedback_merged[feedback_merged['Rating'] <= 2]
        st.dataframe(poor_reviews[['Order_ID', 'Rating', 'Feedback_Text', 'Issue_Category', 'Carrier', 'Route']])

with tab4:
    st.header("ML-Driven Cluster Analysis (Unsupervised Learning)")
    st.write("""
    This tab uses K-Means clustering to discover hidden groups in your data.
    This is an ML model that finds patterns *without* being trained on 'correct' answers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Route Cluster Analysis")

        route_features = ['Distance_KM', 'Total_Cost_INR', 'Traffic_Delay_Minutes']
        route_data_filtered = filtered_data[route_features]

        processed_route_filtered = route_preprocessor.transform(route_data_filtered)
        route_clusters = route_model.predict(processed_route_filtered)

        plot_data_route = filtered_data.copy()
        plot_data_route['Cluster'] = [f"Cluster {c}" for c in route_clusters]

        fig9 = px.scatter(
            plot_data_route,
            x="Distance_KM",
            y="Total_Cost_INR",
            color="Cluster",
            hover_data=["Route", "Traffic_Delay_Minutes"],
            title="Route Clusters (Cost vs. Distance)"
        )
        st.plotly_chart(fig9, use_container_width=True)

    with col2:
        st.subheader("Customer Cluster Analysis")

        customer_data_merged = pd.merge(feedback, data[['Order_ID', 'Order_Value_INR']], on='Order_ID', how='left')
        customer_features = ['Rating', 'Order_Value_INR']
        customer_data = customer_data_merged[customer_features]

        processed_customer_data = customer_preprocessor.transform(customer_data)
        customer_clusters = customer_model.predict(processed_customer_data)

        plot_data_cust = customer_data_merged.copy()
        plot_data_cust['Cluster'] = [f"Cluster {c}" for c in customer_clusters]

        fig10 = px.scatter(
            plot_data_cust,
            x="Order_Value_INR",
            y="Rating",
            color="Cluster",
            hover_data=["Order_ID"],
            title="Customer Clusters (Rating vs. Order Value)"
        )
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")
    st.subheader("Interpreting the Clusters")
    st.write("""
    * **Route Clusters:** Look at the chart. You can now identify groups like "Short, cheap routes" vs. "Long, expensive routes" and see if your costs are appropriate.
    * **Customer Clusters:** This chart helps you find groups like "High-Value, Low-Rating" (at-risk customers) or "Low-Value, High-Rating" (happy but small customers).
    """)

with tab5:
    st.header("Warehouse & Inventory Analysis")

    # Manually create coordinates for the 5 warehouses
    warehouse_locations = {
        'Mumbai': [19.0760, 72.8777],
        'Delhi': [28.7041, 77.1025],
        'Bangalore': [12.9716, 77.5946],
        'Chennai': [13.0827, 80.2707],
        'Kolkata': [22.5726, 88.3639]
    }

    # Add lat/lon to the warehouses dataframe
    # --- FIXED: Use 'Location' column ---
    warehouses['lat'] = warehouses['Location'].map(lambda x: warehouse_locations.get(x, [None])[0])
    warehouses['lon'] = warehouses['Location'].map(lambda x: warehouse_locations.get(x, [None])[1])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Warehouse Location Map")
        st.map(warehouses.dropna(subset=['lat', 'lon']))

        st.subheader("Storage Cost by Warehouse")
        # --- FIXED: Use 'Location' and 'Storage_Cost_per_Unit' ---
        storage_costs_by_warehouse = warehouses.groupby('Location')['Storage_Cost_per_Unit'].sum().reset_index()
        fig11 = px.pie(
            storage_costs_by_warehouse,
            names="Location",
            values="Storage_Cost_per_Unit",
            title="Total Storage Cost Breakdown"
        )
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        st.subheader("Stock Levels by Warehouse")
        # --- FIXED: Use 'Location' and 'Current_Stock_Units' ---
        fig12 = px.bar(
            warehouses,
            x="Location",
            y="Current_Stock_Units",
            color="Product_Category",
            title="Stock Levels by Warehouse & Category",
            barmode='group'
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown("---")
    st.subheader("Warehouse Inventory Data")
    st.dataframe(warehouses)