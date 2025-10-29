# NexGen Logistics - Mission Control Dashboard

This is a comprehensive, 5-tab business intelligence dashboard built for the OFI Services AI Internship Challenge. It provides a holistic 360-degree view of NexGen Logistics' operations, from warehousing to final delivery.

The dashboard is built with Python, Streamlit, Pandas, and Plotly, and includes an unsupervised machine learning (K-Means Clustering) module for advanced customer and route segmentation.

## How to Run

1.  **Prerequisites:** Ensure you have Python 3.9+ installed.

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    Ensure your 7 CSV files (`orders.csv`, `delivery_performance.csv`, etc.) are in the same folder as the `app.py` file.
    ```bash
    streamlit run app.py
    ```

5.  **View the Dashboard:** A new tab will automatically open in your web browser.
