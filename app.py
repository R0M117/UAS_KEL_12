# app.py

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import plotly.graph_objs as go
import plotly.io as pio
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'  # Replace with a secure secret key

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'fintrack.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Setup Logging
file_handler = RotatingFileHandler(os.path.join(LOG_FOLDER, 'fintrack.log'), maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('FinTrack startup')

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database Models
class FinancialData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    month = db.Column(db.Date, unique=True, nullable=False)
    net_savings = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<FinancialData {self.month} - {self.net_savings}>'

class Income(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    source = db.Column(db.String(150), nullable=False)
    amount = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Income {self.date} - {self.source} - {self.amount}>'

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(150), nullable=False)
    amount = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Expense {self.date} - {self.category} - {self.amount}>'

class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    goal = db.Column(db.String(255), nullable=False)
    target_amount = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Goal {self.goal} - {self.target_amount}>'

# Create database tables
with app.app_context():
    db.create_all()

def preprocess_data(df):
    """
    Preprocess the uploaded DataFrame to extract 'month' and 'net_savings'.
    Handles misspellings using fuzzy matching.
    """
    # Define target column names and their standard names
    target_columns = {
        'month': 'month',
        'net saving': 'net_savings'
    }

    # Convert DataFrame columns to lowercase strings for matching
    df_columns = [str(col).lower() for col in df.columns]

    mapped_columns = {}

    for key, standard_name in target_columns.items():
        # Use fuzzy matching to find the best match
        match, score = process.extractOne(key, df_columns)
        if score >= 80:  # Threshold for a good match
            original_col = df.columns[df_columns.index(match)]
            mapped_columns[standard_name] = original_col
        else:
            # If no good match is found, log and return None
            app.logger.warning(f'Could not find a good match for column "{key}". Best match: "{match}" with score {score}.')
            return None

    # Rename the columns to standard names
    df = df.rename(columns={
        mapped_columns['month']: 'month',
        mapped_columns['net_savings']: 'net_savings'
    })

    # Select only the required columns
    df = df[['month', 'net_savings']]

    # Preprocess 'month' to YYYY-MM-DD format
    df['month'] = pd.to_datetime(df['month'], errors='coerce')  # Keep as Timestamp

    # Drop rows with invalid dates
    df = df.dropna(subset=['month'])

    # Convert 'net_savings' to integer
    df['net_savings'] = pd.to_numeric(df['net_savings'], errors='coerce').astype('Int64')

    # Drop rows with invalid net_savings
    df = df.dropna(subset=['net_savings'])

    return df

# Import TheFuzz
from thefuzz import process

def generate_combined_data():
    """
    Generate the combined DataFrame with historical and forecasted net savings.
    Returns combined_df and combined_df_extended.
    """
    # Fetch historical net savings data
    historical_data = FinancialData.query.order_by(FinancialData.month).all()
    if not historical_data:
        return None, None

    data = pd.DataFrame([{'month': entry.month, 'net_savings': entry.net_savings} for entry in historical_data])

    # Convert 'month' to datetime
    data['month'] = pd.to_datetime(data['month'])

    # Handle non-positive net_savings
    if (data['net_savings'] <= 0).any():
        data['net_savings'] = data['net_savings'] + abs(data['net_savings'].min()) + 1

    # Apply Box-Cox transformation
    data['net_savings_boxcox'], lam = boxcox(data['net_savings'])

    # Fit ARIMA model
    stepwise_model = auto_arima(
        data['net_savings_boxcox'],
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=True,
        m=12,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Forecast for next 12 months
    n_periods = 12
    forecast_boxcox = stepwise_model.predict(n_periods=n_periods)
    forecast = inv_boxcox(forecast_boxcox, lam)

    last_date = data['month'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_periods, freq='MS')

    forecast_df = pd.DataFrame({
        'month': future_dates,
        'net_savings': forecast
    })

    # Combine historical and forecast data
    combined_df = pd.concat([data[['month', 'net_savings']], forecast_df], ignore_index=True)
    combined_df['net_savings'] = pd.to_numeric(combined_df['net_savings'], errors='coerce')
    combined_df['accumulated_net_savings'] = combined_df['net_savings'].cumsum()

    # Extended forecast for goal achievement
    n_periods_extended = 720  # Next 720 months (60 years)
    forecast_boxcox_extended = stepwise_model.predict(n_periods=n_periods_extended)
    forecast_boxcox_extended = np.where(forecast_boxcox_extended <= 0, 1e-6, forecast_boxcox_extended)
    forecast_extended = inv_boxcox(forecast_boxcox_extended, lam)
    future_dates_extended = pd.date_range(
        start=combined_df['month'].iloc[-1] + pd.DateOffset(months=1),
        periods=n_periods_extended,
        freq='MS'
    )
    forecast_df_extended = pd.DataFrame({
        'month': future_dates_extended,
        'net_savings': forecast_extended
    })
    combined_df_extended = pd.concat([combined_df[['month', 'net_savings']], forecast_df_extended], ignore_index=True)
    combined_df_extended['net_savings'] = pd.to_numeric(combined_df_extended['net_savings'], errors='coerce')
    combined_df_extended['accumulated_net_savings'] = combined_df_extended['net_savings'].cumsum()

    return combined_df, combined_df_extended

def get_achievement_month(combined_df_extended, target_amount):
    """
    Calculate the achievement month for a given target amount.
    Returns the achievement_month (Timestamp) and achievement_month_str (formatted string).
    """
    achieved = combined_df_extended[combined_df_extended['accumulated_net_savings'] >= target_amount]
    if not achieved.empty:
        achievement_month = achieved.iloc[0]['month']
        achievement_month_str = achievement_month.strftime('%B %Y')
    else:
        achievement_month = None
        achievement_month_str = 'Not achieved within forecast period'
    return achievement_month, achievement_month_str

@app.route('/')
def index():
    """
    Render the main index page.
    """
    return render_template('index.html')

@app.route('/import_data', methods=['GET', 'POST'])
def import_data():
    """
    Handle data import via CSV or Excel files.
    """
    if request.method == 'POST':
        # Check if the form has a file
        if 'file' not in request.files:
            flash('No file part in the request.', 'warning')
            return redirect(request.url)
        
        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            flash('No file selected for uploading.', 'warning')
            return redirect(request.url)

        # If the file is allowed, process it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Read the file into a DataFrame
                if filename.lower().endswith('csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                # Preprocess the DataFrame
                processed_df = preprocess_data(df)

                if processed_df is not None and not processed_df.empty:
                    # Save the processed data to the database
                    for _, row in processed_df.iterrows():
                        # Check if FinancialData for the month already exists
                        financial_entry = FinancialData.query.filter_by(month=row['month'].date()).first()
                        if financial_entry:
                            # Update net_savings
                            financial_entry.net_savings = row['net_savings']
                        else:
                            # Create new FinancialData entry
                            financial_entry = FinancialData(
                                month=row['month'].date(),
                                net_savings=int(row['net_savings'])
                            )
                            db.session.add(financial_entry)
                    db.session.commit()

                    flash('File successfully processed and data imported!', 'success')
                    # Redirect to view_data page
                    return redirect(url_for('view_data'))
                else:
                    flash('Required columns "month" and "net saving" not found or no valid data.', 'warning')
                    return redirect(request.url)

            except Exception as e:
                app.logger.error(f'Error processing file: {e}')
                flash(f'An error occurred while processing the file: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Allowed file types are CSV, XLSX, XLS.', 'warning')
            return redirect(request.url)

    return render_template('import_data.html')

@app.route('/set_income', methods=['GET', 'POST'])
def set_income():
    """
    Handle setting income data.
    """
    if request.method == 'POST':
        date_str = request.form['date']
        source = request.form['source']
        amount = request.form['amount']

        try:
            # Convert date string to date object
            date = pd.to_datetime(date_str).date()

            # Convert amount to integer
            amount = int(amount)

            # Add Income entry
            income_entry = Income(
                date=date,
                source=source,
                amount=amount
            )
            db.session.add(income_entry)

            # Update FinancialData
            financial_entry = FinancialData.query.filter_by(month=date).first()
            if financial_entry:
                financial_entry.net_savings += amount  # Initial net_savings might not consider expenses yet
            else:
                financial_entry = FinancialData(
                    month=date,
                    net_savings=amount
                )
                db.session.add(financial_entry)
            db.session.commit()

            flash('Income successfully added!', 'success')
            return redirect(url_for('view_income'))

        except Exception as e:
            app.logger.error(f'Error adding income: {e}')
            flash(f'An error occurred while adding income: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('set_income.html')

@app.route('/view_income')
def view_income():
    """
    Display all income data.
    """
    try:
        income_data = Income.query.order_by(Income.date).all()
        return render_template('view_income.html', income_data=income_data)
    except Exception as e:
        app.logger.error(f'Error fetching income data: {e}')
        flash(f'An error occurred while fetching income data: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/set_budget', methods=['GET', 'POST'])
def set_budget():
    """
    Handle setting expense data.
    """
    if request.method == 'POST':
        date_str = request.form['date']
        category = request.form['category']
        outcome = request.form['outcome']

        try:
            # Convert date string to date object
            date = pd.to_datetime(date_str).date()

            # Check if income exists for the month
            income_total = db.session.query(db.func.sum(Income.amount)).filter(Income.date == date).scalar()
            if not income_total or income_total == 0:
                flash('Cannot set expense without existing income for the selected month.', 'warning')
                return redirect(request.url)

            # Convert outcome to integer
            outcome = int(outcome)

            # Add Expense entry
            expense_entry = Expense(
                date=date,
                category=category,
                amount=outcome
            )
            db.session.add(expense_entry)

            # Update FinancialData
            financial_entry = FinancialData.query.filter_by(month=date).first()
            if financial_entry:
                financial_entry.net_savings -= outcome
            else:
                # This shouldn't happen as income exists, but handle just in case
                financial_entry = FinancialData(
                    month=date,
                    net_savings=-outcome
                )
                db.session.add(financial_entry)
            db.session.commit()

            flash('Expense successfully added!', 'success')
            return redirect(url_for('view_budget'))

        except Exception as e:
            app.logger.error(f'Error adding expense: {e}')
            flash(f'An error occurred while adding expense: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('set_budget.html')

@app.route('/view_budget')
def view_budget():
    """
    Display all expense data.
    """
    try:
        expense_data = Expense.query.order_by(Expense.date).all()
        return render_template('view_budget.html', expense_data=expense_data)
    except Exception as e:
        app.logger.error(f'Error fetching expense data: {e}')
        flash(f'An error occurred while fetching expense data: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/set_goals', methods=['GET', 'POST'])
def set_goals():
    """
    Handle setting financial goals.
    """
    if request.method == 'POST':
        goal = request.form['goal']
        target = request.form['target']

        try:
            # Convert target to integer
            target_amount = int(target)

            # Add Goal entry
            goal_entry = Goal(
                goal=goal,
                target_amount=target_amount
            )
            db.session.add(goal_entry)
            db.session.commit()

            flash('Goal successfully added!', 'success')
            return redirect(url_for('view_goals'))

        except ValueError:
            flash('Please enter a valid number for the target amount.', 'warning')
            return redirect(request.url)

        except Exception as e:
            app.logger.error(f'Error adding goal: {e}')
            flash(f'An error occurred while adding the goal: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('set_goals.html')

@app.route('/view_goals')
def view_goals():
    """
    Display all financial goals.
    """
    try:
        goals = Goal.query.order_by(Goal.id.desc()).all()
        return render_template('view_goals.html', goals=goals)
    except Exception as e:
        app.logger.error(f'Error fetching goals: {e}')
        flash(f'An error occurred while fetching goals: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Added Route: Delete Individual Goal
@app.route('/delete_goal/<int:goal_id>', methods=['POST'])
def delete_goal(goal_id):
    """
    Delete a Goal entry.
    """
    try:
        goal_entry = Goal.query.get_or_404(goal_id)
        db.session.delete(goal_entry)
        db.session.commit()
        flash('Goal deleted successfully.', 'success')
        return redirect(url_for('view_goals'))
    except Exception as e:
        app.logger.error(f'Error deleting goal: {e}')
        flash(f'An error occurred while deleting the goal: {str(e)}', 'danger')
        return redirect(url_for('view_goals'))

# Added Route: Delete All Goals
@app.route('/delete_all_goals', methods=['POST'])
def delete_all_goals():
    """
    Delete all data from the Goal table.
    """
    try:
        num_goals = Goal.query.delete()
        db.session.commit()
        flash('All goals have been deleted successfully.', 'success')
        return redirect(url_for('view_goals'))
    except Exception as e:
        app.logger.error(f'Error deleting all goals: {e}')
        flash(f'An error occurred while deleting all goals: {str(e)}', 'danger')
        return redirect(url_for('view_goals'))

@app.route('/view_data')
def view_data():
    """
    Display all imported financial data (net savings).
    """
    try:
        data = FinancialData.query.order_by(FinancialData.month).all()
        return render_template('view_data.html', data=data)
    except Exception as e:
        app.logger.error(f'Error fetching financial data: {e}')
        flash(f'An error occurred while fetching financial data: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/delete_data/<int:data_id>', methods=['POST'])
def delete_data(data_id):
    """
    Delete a FinancialData entry and its related Income and Expense data.
    """
    try:
        financial_entry = FinancialData.query.get_or_404(data_id)
        date = financial_entry.month

        # Delete related Income entries
        incomes = Income.query.filter_by(date=date).all()
        for income in incomes:
            db.session.delete(income)

        # Delete related Expense entries
        expenses = Expense.query.filter_by(date=date).all()
        for expense in expenses:
            db.session.delete(expense)

        # Delete FinancialData entry
        db.session.delete(financial_entry)

        db.session.commit()
        flash('Financial data and related income/expense entries deleted successfully.', 'success')
        return redirect(url_for('view_data'))
    except Exception as e:
        app.logger.error(f'Error deleting data: {e}')
        flash(f'An error occurred while deleting data: {str(e)}', 'danger')
        return redirect(url_for('view_data'))

@app.route('/delete_income/<int:income_id>', methods=['POST'])
def delete_income(income_id):
    """
    Delete an Income entry and adjust the corresponding net_savings.
    """
    try:
        income_entry = Income.query.get_or_404(income_id)
        date = income_entry.date
        amount = income_entry.amount

        # Delete the income entry
        db.session.delete(income_entry)

        # Adjust FinancialData
        financial_entry = FinancialData.query.filter_by(month=date).first()
        if financial_entry:
            financial_entry.net_savings -= amount
            if financial_entry.net_savings < 0:
                financial_entry.net_savings = 0  # Prevent negative net_savings
        else:
            # If no FinancialData exists for the month, no adjustment needed
            pass

        db.session.commit()
        flash('Income entry deleted and net savings adjusted successfully.', 'success')
        return redirect(url_for('view_income'))
    except Exception as e:
        app.logger.error(f'Error deleting income: {e}')
        flash(f'An error occurred while deleting income: {str(e)}', 'danger')
        return redirect(url_for('view_income'))

@app.route('/delete_expense/<int:expense_id>', methods=['POST'])
def delete_expense(expense_id):
    """
    Delete an Expense entry and adjust the corresponding net_savings.
    """
    try:
        expense_entry = Expense.query.get_or_404(expense_id)
        date = expense_entry.date
        amount = expense_entry.amount

        # Delete the expense entry
        db.session.delete(expense_entry)

        # Adjust FinancialData
        financial_entry = FinancialData.query.filter_by(month=date).first()
        if financial_entry:
            financial_entry.net_savings += amount
        else:
            # If no FinancialData exists for the month, create one
            financial_entry = FinancialData(
                month=date,
                net_savings=amount
            )
            db.session.add(financial_entry)

        db.session.commit()
        flash('Expense entry deleted and net savings adjusted successfully.', 'success')
        return redirect(url_for('view_budget'))
    except Exception as e:
        app.logger.error(f'Error deleting expense: {e}')
        flash(f'An error occurred while deleting expense: {str(e)}', 'danger')
        return redirect(url_for('view_budget'))

@app.route('/delete_all_data', methods=['POST'])
def delete_all_data():
    """
    Delete all data from FinancialData, Income, and Expense tables.
    """
    try:
        # Delete all Income entries
        num_incomes = Income.query.delete()

        # Delete all Expense entries
        num_expenses = Expense.query.delete()

        # Delete all FinancialData entries
        num_financial = FinancialData.query.delete()

        db.session.commit()
        flash('All financial data, incomes, and expenses have been deleted successfully.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f'Error deleting all data: {e}')
        flash(f'An error occurred while deleting all data: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/report')
def report_view():
    """
    Generate and display the financial report with forecasts and goal achievements.
    """
    try:
        # Generate combined data
        combined_df, combined_df_extended = generate_combined_data()
        if combined_df is None or combined_df_extended is None:
            flash('No financial data available to generate a report.', 'warning')
            return redirect(url_for('index'))

        # Prepare forecast table
        forecast_df = combined_df[combined_df['month'] > combined_df['month'].iloc[-13]].copy()
        forecast_df = forecast_df.iloc[1:]  # Exclude the last historical data point

        # Fetch all goals
        goals = Goal.query.all()
        if not goals:
            flash('No financial goals set. Please set your goals to see goal achievements.', 'warning')
            goal_achievements_df = pd.DataFrame(columns=['Goal', 'Target Amount (IDR)', 'Achievement Month'])
        else:
            goal_achievements = []
            for goal in goals:
                goal_name = goal.goal
                target_amount = goal.target_amount

                achievement_month, achievement_month_str = get_achievement_month(combined_df_extended, target_amount)

                goal_achievements.append({
                    'Goal': goal_name,
                    'Target Amount (IDR)': "{:,}".format(target_amount),
                    'Achievement Month': achievement_month_str
                })

            goal_achievements_df = pd.DataFrame(goal_achievements)

        # Create the forecast plot using Plotly
        fig = go.Figure()

        # Historical Net Savings
        historical_data = combined_df[combined_df['month'] <= combined_df['month'].iloc[-13]]
        fig.add_trace(go.Scatter(
            x=historical_data['month'],
            y=historical_data['net_savings'],
            name='Historical Net Savings',
            line=dict(color='blue')
        ))

        # Forecasted Net Savings
        forecast_data = combined_df[combined_df['month'] > combined_df['month'].iloc[-13]]
        fig.add_trace(go.Scatter(
            x=forecast_data['month'],
            y=forecast_data['net_savings'],
            name='Forecasted Net Savings',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            template="simple_white",
            font=dict(size=18),
            title_text='Net Savings Forecast',
            # Removed fixed width and height for responsiveness
            autosize=True,
            title_x=0.5,
            # height=600,  # Removed fixed height
            xaxis_title='Date',
            yaxis_title='Net Savings (IDR)',
            legend=dict(x=0, y=1.1, orientation='h')  # Position legend above the graph
        )

        # Enable responsive Plotly graph
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        forecast_plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'responsive': True})

        # Render the report template
        return render_template('report.html',
                               forecast_plot=forecast_plot,
                               forecast_table=forecast_df,
                               goal_achievements=goal_achievements_df)
    except Exception as e:
        app.logger.error(f'Error generating report: {e}')
        flash(f'An error occurred while generating the report: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Modified Route: Goal Progression Plot
@app.route('/goal_progress/<string:goal_name>')
def goal_progress(goal_name):
    """
    Generate and display the goal progression plot for a specific goal.
    """
    try:
        # Fetch the specified goal from the database
        goal = Goal.query.filter_by(goal=goal_name).first()
        if not goal:
            flash(f'Goal "{goal_name}" not found.', 'warning')
            return redirect(url_for('view_goals'))

        target_amount = goal.target_amount

        # Generate combined data
        combined_df, combined_df_extended = generate_combined_data()
        if combined_df is None or combined_df_extended is None:
            flash('No financial data available to generate the goal progression.', 'warning')
            return redirect(url_for('index'))

        # Get achievement month
        achievement_month, achievement_month_str = get_achievement_month(combined_df_extended, target_amount)

        if achievement_month is not None:
            # Truncate the data up to the achievement month
            plot_data = combined_df_extended[combined_df_extended['month'] <= achievement_month]
        else:
            # If goal not achieved within forecast, use all data
            plot_data = combined_df_extended.copy()

        # Create the goal progression plot
        fig_goal = go.Figure()
        fig_goal.add_trace(go.Scatter(
            x=plot_data['month'],
            y=plot_data['accumulated_net_savings'],
            mode='lines',
            name='Accumulated Net Savings',
            line=dict(color='blue')
        ))
        fig_goal.add_trace(go.Scatter(
            x=plot_data['month'],
            y=[target_amount] * len(plot_data),
            mode='lines',
            name=f'Target Amount ({goal_name})',
            line=dict(color='red', dash='dash')
        ))
        if achievement_month is not None:
            achievement_value = plot_data[plot_data['month'] == achievement_month]['accumulated_net_savings'].values[0]
            fig_goal.add_trace(go.Scatter(
                x=[achievement_month],
                y=[achievement_value],
                mode='markers',
                name='Goal Achieved',
                marker=dict(color='green', size=10, symbol='star')
            ))

        fig_goal.update_layout(
            template="simple_white",
            font=dict(size=18),
            title_text=f'Goal Progression: {goal_name}',
            # Removed fixed width and height for responsiveness
            autosize=True,
            title_x=0.5,
            # height=600,  # Removed fixed height
            xaxis_title='Date',
            yaxis_title='Accumulated Net Savings (IDR)',
            legend=dict(x=0, y=1.1, orientation='h')  # Position legend above the graph
        )

        # Enable responsive Plotly graph
        fig_goal.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        plot = pio.to_html(fig_goal, full_html=False, include_plotlyjs='cdn', config={'responsive': True})

        return render_template('goal_progress.html',
                               goal_name=goal_name,
                               plot=plot)
    except Exception as e:
        app.logger.error(f'Error generating goal progress for {goal_name}: {e}')
        flash(f'An error occurred while generating the goal progress: {str(e)}', 'danger')
        return redirect(url_for('view_goals'))

if __name__ == '__main__':
    app.run(debug=True)
