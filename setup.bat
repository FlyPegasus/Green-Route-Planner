echo "Creating Virtual Environment"
python -m venv GreenRoutePlanner
cd GreenRoutePlanner

echo "Activating Virtual Environment"
Scripts\activate

echo "Installing dependencies"
pip install flask