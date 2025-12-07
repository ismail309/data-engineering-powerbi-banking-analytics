"""
Main banking data generator using Faker
"""

# Add this as the FIRST line in your file
import sys
import io

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BankDataGenerator:
    """Generate realistic banking data for Power BI dashboards"""
    
    def __init__(self, seed: int = 42, country: str = 'en_US'):
        """
        Initialize generator
        
        Args:
            seed: Random seed for reproducibility
            country: Country code for localized data
        """
        self.faker = Faker(country)
        self.faker.seed_instance(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        
        # Configuration
        self.config = {
            'min_customers': 1000,
            'max_customers': 10000,
            'accounts_per_customer': (1, 4),
            'transactions_per_account': (10, 100),
            'num_branches': 50,
            'loan_approval_rate': 0.3  # 30% of customers get loans
        }
        
        logger.info(f"Initialized BankDataGenerator with seed={seed}")
    
    def generate_customers(self, num_customers: int = 5000) -> pd.DataFrame:
        """Generate customer demographic data"""
        logger.info(f"Generating {num_customers} customers...")
        
        customers = []
        for i in range(num_customers):
            # Customer tier based on income
            income_tier = np.random.choice(
                ['Basic', 'Standard', 'Premium', 'Private'],
                p=[0.4, 0.35, 0.2, 0.05]
            )
            
            # Generate realistic age distribution
            age = int(np.random.normal(45, 15))
            age = max(18, min(90, age))
            
            customer = {
                'customer_id': f'CUST{100000 + i:06d}',
                'first_name': self.faker.first_name(),
                'last_name': self.faker.last_name(),
                'full_name': '',  # Will fill later
                'email': f'customer.{100000 + i}@example.com',
                'phone': self.faker.phone_number(),
                'address': self.faker.street_address(),
                'city': self.faker.city(),
                'state': self.faker.state_abbr(),
                'zip_code': self.faker.zipcode(),
                'country': 'US',
                'date_of_birth': self.faker.date_of_birth(minimum_age=18, maximum_age=90),
                'age': age,
                'gender': np.random.choice(['M', 'F'], p=[0.49, 0.51]),
                'occupation': self.faker.job(),
                'annual_income': self._generate_income(income_tier),
                'customer_tier': income_tier,
                'credit_score': int(np.random.normal(720, 60)),
                'credit_score_category': '',
                'customer_since': self.faker.date_between(start_date='-10y', end_date='-6m'),
                'preferred_branch_id': f'BR{random.randint(1, self.config["num_branches"]):03d}',
                'risk_score': np.random.randint(1, 100),
                'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                                   p=[0.3, 0.5, 0.15, 0.05]),
                'dependents': np.random.poisson(1.5),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                                    p=[0.3, 0.4, 0.25, 0.05]),
                'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], 
                                                      p=[0.7, 0.15, 0.1, 0.05]),
                'marketing_consent': np.random.choice([True, False], p=[0.7, 0.3]),
                'is_active': True,
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
            
            # Fill derived fields
            customer['full_name'] = f"{customer['first_name']} {customer['last_name']}"
            customer['credit_score'] = max(300, min(850, customer['credit_score']))
            customer['credit_score_category'] = self._get_credit_score_category(customer['credit_score'])
            
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df)} customers")
        return df
    
    def _generate_income(self, tier: str) -> float:
        """Generate income based on customer tier"""
        if tier == 'Basic':
            return round(np.random.uniform(20000, 50000), 2)
        elif tier == 'Standard':
            return round(np.random.uniform(50000, 100000), 2)
        elif tier == 'Premium':
            return round(np.random.uniform(100000, 250000), 2)
        else:  # Private
            return round(np.random.uniform(250000, 1000000), 2)
    
    def _get_credit_score_category(self, score: int) -> str:
        """Categorize credit score"""
        if score >= 800:
            return 'Excellent'
        elif score >= 740:
            return 'Very Good'
        elif score >= 670:
            return 'Good'
        elif score >= 580:
            return 'Fair'
        else:
            return 'Poor'
    
    def generate_accounts(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate account data for customers"""
        logger.info("Generating accounts...")
        
        accounts = []
        account_counter = 5000000
        
        for _, customer in customers_df.iterrows():
            # Determine number of accounts for this customer
            num_accounts = np.random.randint(
                self.config['accounts_per_customer'][0],
                self.config['accounts_per_customer'][1] + 1
            )
            
            for i in range(num_accounts):
                # Account type with realistic distribution
                account_type = np.random.choice(
                    ['Checking', 'Savings', 'Money Market', 'CD', 'IRA', 'Brokerage'],
                    p=[0.35, 0.30, 0.15, 0.10, 0.05, 0.05]
                )
                
                # Generate account number
                account_number = f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
                
                # Generate realistic balance based on account type and customer income
                base_balance = customer['annual_income'] / np.random.uniform(5, 20)
                if account_type == 'Checking':
                    balance = np.random.exponential(base_balance * 0.1)
                elif account_type == 'Savings':
                    balance = np.random.exponential(base_balance * 0.3)
                elif account_type in ['Money Market', 'CD']:
                    balance = np.random.exponential(base_balance * 0.5)
                else:
                    balance = np.random.exponential(base_balance * 0.8)
                
                balance = round(max(100, balance), 2)
                
                # Interest rate based on account type and customer tier
                if account_type in ['Savings', 'Money Market', 'CD']:
                    base_rate = 0.02
                    if customer['customer_tier'] == 'Premium':
                        base_rate += 0.005
                    elif customer['customer_tier'] == 'Private':
                        base_rate += 0.01
                    interest_rate = round(base_rate + np.random.uniform(-0.005, 0.005), 4)
                else:
                    interest_rate = 0.0
                
                account = {
                    'account_id': f'ACC{account_counter}',
                    'customer_id': customer['customer_id'],
                    'account_number': account_number,
                    'account_type': account_type,
                    'account_subtype': self._get_account_subtype(account_type),
                    'balance': balance,
                    'available_balance': round(balance * np.random.uniform(0.95, 1.0), 2),
                    'currency': 'USD',
                    'interest_rate': interest_rate,
                    'open_date': self.faker.date_between(
                        start_date=customer['customer_since'],
                        end_date='today'
                    ),
                    'account_status': np.random.choice(['Active', 'Dormant', 'Closed'], 
                                                       p=[0.85, 0.10, 0.05]),
                    'overdraft_limit': 1000 if account_type == 'Checking' else 0,
                    'minimum_balance_requirement': 100 if account_type in ['Savings', 'Money Market'] else 0,
                    'monthly_fee': 0 if customer['customer_tier'] in ['Premium', 'Private'] else np.random.choice([0, 5, 10], p=[0.7, 0.2, 0.1]),
                    'branch_id': customer['preferred_branch_id'],
                    'last_transaction_date': None,  # Will be updated with transactions
                    'created_at': datetime.now(),
                    'last_updated': datetime.now()
                }
                
                accounts.append(account)
                account_counter += 1
        
        df = pd.DataFrame(accounts)
        logger.info(f"Generated {len(df)} accounts")
        return df
    
    def _get_account_subtype(self, account_type: str) -> str:
        """Get account subtype"""
        subtypes = {
            'Checking': ['Personal Checking', 'Business Checking', 'Student Checking'],
            'Savings': ['Regular Savings', 'High-Yield Savings', 'Youth Savings'],
            'Money Market': ['Money Market Deposit', 'Premium Money Market'],
            'CD': ['6-month CD', '1-year CD', '5-year CD'],
            'IRA': ['Traditional IRA', 'Roth IRA'],
            'Brokerage': ['Standard Brokerage', 'Retirement Brokerage']
        }
        return np.random.choice(subtypes.get(account_type, ['Standard']))
    
    def generate_transactions(self, accounts_df: pd.DataFrame, 
                              start_date: str = '2023-01-01',
                              end_date: str = '2024-12-31') -> pd.DataFrame:
        """Generate transaction data"""
        logger.info("Generating transactions...")
        
        transactions = []
        transaction_counter = 900000000
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Transaction type probabilities
        tx_types = {
            'DEPOSIT': 0.25,
            'WITHDRAWAL': 0.35,
            'TRANSFER': 0.20,
            'PAYMENT': 0.15,
            'FEE': 0.03,
            'INTEREST': 0.02
        }
        
        tx_type_list = list(tx_types.keys())
        tx_probs = list(tx_types.values())
        
        for _, account in accounts_df.iterrows():
            if account['account_status'] != 'Active':
                continue
            
            # Generate random number of transactions for this account
            num_transactions = np.random.randint(
                self.config['transactions_per_account'][0],
                self.config['transactions_per_account'][1] + 1
            )
            
            for _ in range(num_transactions):
                # Generate transaction date (more recent = more frequent)
                days_range = (end_dt - start_dt).days
                days_ago = np.random.exponential(days_range * 0.3)
                transaction_date = end_dt - timedelta(days=min(days_range, int(days_ago)))
                
                # Generate transaction time (business hours more common)
                hour = int(np.random.normal(14, 4))  # Peak around 2 PM
                hour = max(0, min(23, hour))
                minute = random.randint(0, 59)
                
                transaction_type = np.random.choice(tx_type_list, p=tx_probs)
                
                # Generate amount based on transaction type
                amount = self._generate_transaction_amount(transaction_type, account['account_type'])
                
                transaction = {
                    'transaction_id': f'TXN{transaction_counter}',
                    'account_id': account['account_id'],
                    'transaction_date': transaction_date.date(),
                    'transaction_time': f"{hour:02d}:{minute:02d}:00",
                    'transaction_datetime': datetime.combine(transaction_date.date(), 
                                                           datetime.min.time().replace(hour=hour, minute=minute)),
                    'transaction_type': transaction_type,
                    'amount': amount,
                    'description': self._generate_transaction_description(transaction_type),
                    'merchant_name': self.faker.company() if transaction_type in ['PAYMENT', 'WITHDRAWAL'] else None,
                    'merchant_category': self._get_merchant_category() if transaction_type in ['PAYMENT', 'WITHDRAWAL'] else None,
                    'merchant_id': f'MER{random.randint(10000, 99999)}' if transaction_type in ['PAYMENT', 'WITHDRAWAL'] else None,
                    'channel': np.random.choice(['Branch', 'ATM', 'Online', 'Mobile', 'Phone'], 
                                                p=[0.15, 0.25, 0.35, 0.20, 0.05]),
                    'status': np.random.choice(['Completed', 'Pending', 'Failed'], 
                                               p=[0.95, 0.04, 0.01]),
                    'location': f"{self.faker.city()}, {self.faker.state_abbr()}" if np.random.random() > 0.3 else 'Online',
                    'reference_number': f'REF{random.randint(100000000, 999999999)}',
                    'transaction_category': self._categorize_transaction(transaction_type),
                    'is_fraudulent': np.random.choice([True, False], p=[0.001, 0.999]),  # 0.1% fraud rate
                    'created_at': datetime.now()
                }
                
                transactions.append(transaction)
                transaction_counter += 1
        
        df = pd.DataFrame(transactions)
        
        # Sort by date
        df = df.sort_values(['transaction_datetime'])
        logger.info(f"Generated {len(df)} transactions")
        return df
    
    def _generate_transaction_amount(self, tx_type: str, account_type: str) -> float:
        """Generate realistic transaction amount"""
        if tx_type == 'DEPOSIT':
            base = np.random.exponential(1000)
        elif tx_type == 'WITHDRAWAL':
            base = np.random.exponential(200)
        elif tx_type == 'TRANSFER':
            base = np.random.exponential(500)
        elif tx_type == 'PAYMENT':
            base = np.random.exponential(150)
        elif tx_type == 'FEE':
            base = -np.random.choice([5, 10, 25, 50, 100])
        else:  # INTEREST
            base = np.random.exponential(10)
        
        # Adjust for account type
        if account_type in ['Money Market', 'CD', 'IRA', 'Brokerage']:
            base *= np.random.uniform(2, 5)
        
        return round(base, 2)
    
    def _generate_transaction_description(self, tx_type: str) -> str:
        """Generate transaction description"""
        descriptions = {
            'DEPOSIT': ['Salary Deposit', 'Cash Deposit', 'Check Deposit', 
                       'Transfer Received', 'Interest Payment', 'Dividend'],
            'WITHDRAWAL': ['ATM Withdrawal', 'Cash Withdrawal', 'Teller Withdrawal'],
            'TRANSFER': ['Fund Transfer', 'Account Transfer', 'Wire Transfer',
                        'Bill Payment', 'Peer-to-Peer Transfer'],
            'PAYMENT': ['Credit Card Payment', 'Mortgage Payment', 'Utility Bill',
                       'Online Purchase', 'Subscription', 'Insurance'],
            'FEE': ['Monthly Fee', 'ATM Fee', 'Overdraft Fee', 
                   'Wire Fee', 'Late Fee'],
            'INTEREST': ['Interest Credited', 'Dividend Payment']
        }
        return np.random.choice(descriptions.get(tx_type, ['Transaction']))
    
    def _get_merchant_category(self) -> str:
        """Get merchant category"""
        categories = [
            'Retail', 'Groceries', 'Restaurant', 'Entertainment',
            'Utilities', 'Healthcare', 'Transportation', 'Online Retail',
            'Gas/Automotive', 'Travel', 'Education', 'Insurance'
        ]
        return np.random.choice(categories)
    
    def _categorize_transaction(self, tx_type: str) -> str:
        """Categorize transaction for reporting"""
        categories = {
            'DEPOSIT': 'Income',
            'WITHDRAWAL': 'Cash',
            'TRANSFER': 'Transfer',
            'PAYMENT': 'Expense',
            'FEE': 'Bank Fees',
            'INTEREST': 'Income'
        }
        return categories.get(tx_type, 'Other')
    
    def generate_branches(self) -> pd.DataFrame:
        """Generate branch data"""
        logger.info("Generating branches...")
        
        branches = []
        states = ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        
        for i in range(1, self.config['num_branches'] + 1):
            state = np.random.choice(states)
            
            branch = {
                'branch_id': f'BR{i:03d}',
                'branch_name': f'{self.faker.city()} Branch',
                'branch_code': f'BC{i:04d}',
                'address': self.faker.street_address(),
                'city': self.faker.city(),
                'state': state,
                'zip_code': self.faker.zipcode(),
                'phone': self.faker.phone_number(),
                'email': f'branch{i}@bank.com',
                'manager_name': self.faker.name(),
                'manager_id': f'MGR{random.randint(1000, 9999)}',
                'open_date': self.faker.date_between(start_date='-20y', end_date='-1y'),
                'total_deposits': round(np.random.exponential(50000000), 2),
                'total_loans': round(np.random.exponential(30000000), 2),
                'num_employees': np.random.randint(5, 50),
                'region': self._get_region(state),
                'branch_type': np.random.choice(['Full Service', 'Express', 'Commercial'], 
                                                p=[0.6, 0.3, 0.1]),
                'operating_hours': '9:00 AM - 5:00 PM',
                'has_atm': np.random.choice([True, False], p=[0.8, 0.2]),
                'has_drive_thru': np.random.choice([True, False], p=[0.4, 0.6]),
                'square_footage': np.random.randint(2000, 10000),
                'created_at': datetime.now()
            }
            branches.append(branch)
        
        df = pd.DataFrame(branches)
        logger.info(f"Generated {len(df)} branches")
        return df
    
    def _get_region(self, state: str) -> str:
        """Get region from state"""
        regions = {
            'Northeast': ['NY', 'PA', 'MA', 'NJ', 'CT'],
            'West': ['CA', 'WA', 'OR', 'CO', 'AZ'],
            'South': ['TX', 'FL', 'GA', 'NC', 'TN'],
            'Midwest': ['IL', 'OH', 'MI', 'IN', 'MO']
        }
        
        for region, states_in_region in regions.items():
            if state in states_in_region:
                return region
        return 'Other'
    
    def generate_loans(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate loan data"""
        logger.info("Generating loans...")
        
        # Select customers eligible for loans (based on credit score)
        eligible_customers = customers_df[customers_df['credit_score'] > 650].copy()
        num_loans = int(len(eligible_customers) * self.config['loan_approval_rate'])
        
        if num_loans == 0:
            return pd.DataFrame()
        
        # Sample customers for loans
        loan_customers = eligible_customers.sample(n=num_loans, random_state=self.seed)
        
        loans = []
        loan_counter = 7000000
        
        for i, (_, customer) in enumerate(loan_customers.iterrows()):
            # Loan type distribution
            loan_type = np.random.choice(
                ['Personal Loan', 'Auto Loan', 'Home Mortgage', 'Education Loan', 'Business Loan'],
                p=[0.4, 0.25, 0.20, 0.10, 0.05]
            )
            
            # Loan amount based on type and income
            if loan_type == 'Personal Loan':
                base_amount = customer['annual_income'] * 0.5
            elif loan_type == 'Auto Loan':
                base_amount = customer['annual_income'] * 0.8
            elif loan_type == 'Home Mortgage':
                base_amount = customer['annual_income'] * 3
            elif loan_type == 'Education Loan':
                base_amount = customer['annual_income'] * 0.3
            else:  # Business Loan
                base_amount = customer['annual_income'] * 2
            
            # Add randomness
            loan_amount = base_amount * np.random.uniform(0.7, 1.3)
            loan_amount = round(max(1000, loan_amount), 2)
            
            # Interest rate based on credit score
            base_rate = 5.0  # Base for excellent credit
            credit_adjustment = (850 - customer['credit_score']) / 850 * 8  # Up to 8% additional
            interest_rate = round(base_rate + credit_adjustment + np.random.uniform(-0.5, 0.5), 2)
            
            # Tenure based on loan type
            if loan_type == 'Personal Loan':
                tenure = np.random.choice([12, 24, 36, 48, 60])
            elif loan_type == 'Auto Loan':
                tenure = np.random.choice([36, 48, 60, 72])
            elif loan_type == 'Home Mortgage':
                tenure = np.random.choice([180, 240, 360])  # 15, 20, 30 years
            elif loan_type == 'Education Loan':
                tenure = np.random.choice([60, 120, 180])
            else:  # Business Loan
                tenure = np.random.choice([60, 120, 180])
            
            # Calculate EMI
            monthly_rate = interest_rate / 1200
            emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
            emi = round(emi, 2)
            
            # Determine loan status
            status_probs = {
                'Active': 0.85,
                'Delinquent (30-60 days)': 0.05,
                'Delinquent (60-90 days)': 0.03,
                'Default': 0.02,
                'Paid Off': 0.05
            }
            status = np.random.choice(list(status_probs.keys()), p=list(status_probs.values()))
            
            # Calculate remaining balance
            if status == 'Paid Off':
                remaining_balance = 0
            elif status == 'Default':
                remaining_balance = round(loan_amount * np.random.uniform(0.3, 0.8), 2)
            else:
                # Active loan - random remaining balance
                payments_made = np.random.randint(1, tenure)
                remaining_balance = round(emi * (tenure - payments_made), 2)
            
            loan = {
                'loan_id': f'LOAN{loan_counter}',
                'customer_id': customer['customer_id'],
                'loan_type': loan_type,
                'loan_purpose': self._get_loan_purpose(loan_type),
                'disbursement_date': self.faker.date_between(
                    start_date=customer['customer_since'],
                    end_date='-6m'
                ),
                'loan_amount': loan_amount,
                'interest_rate': interest_rate,
                'tenure_months': tenure,
                'emi': emi,
                'remaining_balance': remaining_balance,
                'next_due_date': self.faker.date_between(start_date='today', end_date='+30d'),
                'loan_status': status,
                'default_probability': round(np.random.uniform(0, 0.5), 3),
                'collateral_value': round(loan_amount * np.random.uniform(1.0, 1.5), 2) if loan_type in ['Auto Loan', 'Home Mortgage'] else 0,
                'collateral_type': 'Vehicle' if loan_type == 'Auto Loan' else 'Property' if loan_type == 'Home Mortgage' else None,
                'branch_id': customer['preferred_branch_id'],
                'loan_officer': self.faker.name(),
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
            
            loans.append(loan)
            loan_counter += 1
        
        df = pd.DataFrame(loans)
        logger.info(f"Generated {len(df)} loans")
        return df
    
    def _get_loan_purpose(self, loan_type: str) -> str:
        """Get loan purpose based on type"""
        purposes = {
            'Personal Loan': ['Debt Consolidation', 'Home Improvement', 'Medical Expenses', 'Vacation', 'Wedding'],
            'Auto Loan': ['New Car', 'Used Car', 'Motorcycle', 'RV'],
            'Home Mortgage': ['Purchase', 'Refinance', 'Construction'],
            'Education Loan': ['Undergraduate', 'Graduate', 'Vocational Training'],
            'Business Loan': ['Startup', 'Expansion', 'Equipment', 'Working Capital']
        }
        return np.random.choice(purposes.get(loan_type, ['General']))
    
    def generate_all_data(self, num_customers: int = 5000) -> Dict[str, pd.DataFrame]:
        """Generate complete banking dataset"""
        logger.info(f"Starting complete data generation for {num_customers} customers...")
        
        # Generate data
        customers = self.generate_customers(num_customers)
        accounts = self.generate_accounts(customers)
        transactions = self.generate_transactions(accounts)
        branches = self.generate_branches()
        loans = self.generate_loans(customers)
        
        # Update account last transaction date
        if not transactions.empty:
            last_transactions = transactions.groupby('account_id')['transaction_datetime'].max().reset_index()
            last_transactions.columns = ['account_id', 'last_transaction_date']
            accounts = accounts.merge(last_transactions, on='account_id', how='left')
        
        data_dict = {
            'customers': customers,
            'accounts': accounts,
            'transactions': transactions,
            'branches': branches,
            'loans': loans
        }
        
        logger.info("✅ Data generation complete!")
        self._print_summary(data_dict)
        
        return data_dict
    
    def _print_summary(self, data_dict: Dict[str, pd.DataFrame]):
        """Print data generation summary"""
        print("\n" + "="*60)
        print("DATA GENERATION SUMMARY")
        print("="*60)
        
        total_records = 0
        for name, df in data_dict.items():
            if not df.empty:
                record_count = len(df)
                total_records += record_count
                print(f"{name.capitalize():<15} {record_count:>10,} records")
        
        print("-"*60)
        print(f"{'Total':<15} {total_records:>10,} records")
        
        # Calculate some metrics
        if 'customers' in data_dict:
            customers = data_dict['customers']
            print(f"\nCustomer Metrics:")
            print(f"  • Avg Credit Score: {customers['credit_score'].mean():.0f}")
            print(f"  • Avg Annual Income: ${customers['annual_income'].mean():,.0f}")
            print(f"  • Customer Tiers: {customers['customer_tier'].value_counts().to_dict()}")
        
        if 'accounts' in data_dict:
            accounts = data_dict['accounts']
            print(f"\nAccount Metrics:")
            print(f"  • Total Balance: ${accounts['balance'].sum():,.2f}")
            print(f"  • Avg Balance: ${accounts['balance'].mean():,.2f}")
            print(f"  • Account Types: {accounts['account_type'].value_counts().to_dict()}")
        
        print("="*60)
    
    def save_to_files(self, data_dict: Dict[str, pd.DataFrame], 
                      output_dir: str = 'data',
                      formats: List[str] = ['csv', 'parquet']):
        """Save generated data to files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            if df.empty:
                continue
                
            # Save in requested formats
            for fmt in formats:
                if fmt == 'csv':
                    filepath = os.path.join(output_dir, f'{name}.csv')
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {len(df)} rows to {filepath}")
                
                elif fmt == 'parquet':
                    filepath = os.path.join(output_dir, f'{name}.parquet')
                    df.to_parquet(filepath, index=False)
                    logger.info(f"Saved {len(df)} rows to {filepath}")
                
                elif fmt == 'excel':
                    # Save sample to Excel (full dataset might be too large)
                    sample_size = min(10000, len(df))
                    sample_df = df.head(sample_size)
                    filepath = os.path.join(output_dir, f'{name}_sample.xlsx')
                    sample_df.to_excel(filepath, index=False, sheet_name=name[:31])
                    logger.info(f"Saved {sample_size} sample rows to {filepath}")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'seed': self.seed,
            'record_counts': {name: len(df) for name, df in data_dict.items()}
        }
        
        metadata_path = os.path.join(output_dir, 'generation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main function to run data generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate banking data for Power BI dashboards')
    parser.add_argument('--customers', type=int, default=5000,
                       help='Number of customers to generate (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'parquet', 'both'],
                       help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BANKING DATA GENERATOR FOR POWER BI")
    print("="*60)
    
    # Initialize generator
    generator = BankDataGenerator(seed=args.seed)
    
    # Determine formats
    if args.format == 'both':
        formats = ['csv', 'parquet']
    else:
        formats = [args.format]
    
    try:
        # Generate data
        data_dict = generator.generate_all_data(num_customers=args.customers)
        
        # Save to files
        generator.save_to_files(data_dict, output_dir=args.output, formats=formats)
        
        print(f"\n✅ Data generation successful!")
        print(f"📁 Files saved in: {args.output}/")
        print(f"🔧 Seed used: {args.seed} (for reproducibility)")
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise

if __name__ == "__main__":
    main()