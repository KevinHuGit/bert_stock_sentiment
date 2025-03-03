import requests
import pandas as pd
import numpy as np
import datetime as dt
from pysabr import black
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Any, Optional, Tuple
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

class SentimentEnhancedOptionsPricing:
    def __init__(self, 
                 ticker: str, 
                 marketaux_api_key: str,
                 bert_model: str = 'bert-base-uncased',
                 use_pretrained_sentiment_model: bool = False,
                 finetune_on_marketaux: bool = True,
                 finetune_epochs: int = 3):
        """
        Initialize the sentiment-enhanced options pricing model.
        
        Args:
            ticker: Stock ticker symbol
            marketaux_api_key: API key for Marketaux news API
            bert_model: Pre-trained BERT model to use
            use_pretrained_sentiment_model: Whether to use a pre-trained sentiment model (if available)
            finetune_on_marketaux: Whether to fine-tune the BERT model on Marketaux sentiment data
            finetune_epochs: Number of epochs for fine-tuning (if enabled)
        """
        self.ticker = ticker
        self.marketaux_api_key = marketaux_api_key
        self.finetune_epochs = finetune_epochs
        
        # Set up BERT model for sentiment analysis
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            if use_pretrained_sentiment_model:
                # Use a pre-trained sentiment model (if this fails, fall back to base BERT)
                print("Trying to load pre-trained financial sentiment model...")
                self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
                self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert').to(self.device)
                print("Successfully loaded pre-trained financial sentiment model.")
            else:
                # Use base BERT model
                print("Loading base BERT model...")
                self.tokenizer = BertTokenizer.from_pretrained(bert_model)
                self.model = BertForSequenceClassification.from_pretrained(
                    bert_model, 
                    num_labels=3  # Negative, Neutral, Positive
                ).to(self.device)
                
                # Note about initialization warning
                print("Note: The classifier layers are initialized randomly. This is expected behavior.")
        except Exception as e:
            print(f"Error loading BERT model: {e}. Initializing simple sentiment analysis.")
            # Fall back to simple sentiment analysis without BERT
            self.tokenizer = None
            self.model = None
        
        # Generate some dummy stock data instead of fetching from yfinance
        self.stock_data = self._generate_dummy_stock_data()
        
        # Cache for news and sentiment to avoid redundant API calls
        self.news_cache = {}
        self.sentiment_cache = {}
        
        # Fine-tune the model on Marketaux data if requested
        if self.model is not None and finetune_on_marketaux:
            self.finetune_model_on_marketaux()
        else:
            if self.model is not None:
                self.model.eval()  # Set model to evaluation mode
        
    def _generate_dummy_stock_data(self) -> pd.DataFrame:
        """
        Generate dummy stock data instead of fetching from yfinance
        
        Returns:
            DataFrame containing simulated historical stock data
        """
        # Create dummy data
        dates = pd.date_range(end=dt.datetime.now(), periods=252)
        
        # Get a reasonable starting price for dummy data
        price_map = {
            'AAPL': 175.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'AMZN': 180.0, 'META': 480.0,
            'TSLA': 175.0, 'NVDA': 850.0, 'JPM': 190.0, 'V': 275.0, 'WMT': 60.0,
            'JNJ': 145.0, 'PG': 160.0, 'BAC': 35.0, 'DIS': 110.0, 'NFLX': 600.0
        }
        
        base_price = price_map.get(self.ticker, 100.0)
        end_price = base_price * (1 + np.random.uniform(-0.3, 0.5))  # Random trend
        
        # Add some volatility to make it more realistic
        prices = np.linspace(base_price, end_price, 252)
        # Add some noise to the price series
        noise = np.random.normal(0, base_price * 0.02, 252)  # 2% daily volatility
        prices = prices + noise
        prices = np.maximum(prices, base_price * 0.5)  # Ensure prices don't go too low
        
        hist_data = pd.DataFrame(
            index=dates,
            data={
                'Open': prices,
                'High': prices * (1 + np.random.uniform(0, 0.02, 252)),  # Up to 2% higher
                'Low': prices * (1 - np.random.uniform(0, 0.02, 252)),   # Up to 2% lower
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, 252)
            }
        )
        
        print(f"Generated dummy stock data for {self.ticker}")
        return hist_data
    
    def fetch_news(self, days_back: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news articles for the ticker from Marketaux API.
        
        Args:
            days_back: Number of days to look back for news
            limit: Maximum number of articles to fetch
            
        Returns:
            List of news articles with title, description, and published date
        """
        # Check if we have cached news for this request
        cache_key = f"{self.ticker}_{days_back}_{limit}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]
        
        # Calculate date range
        end_date = dt.datetime.now().strftime('%Y-%m-%d')
        start_date = (dt.datetime.now() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Marketaux API endpoint
        url = "https://api.marketaux.com/v1/news/all"
        
        # Parameters for the API request
        params = {
            'api_token': self.marketaux_api_key,
            'symbols': self.ticker,
            'countries': 'us',
            'filter_entities': 'true',
            'limit': limit,  # Limit to specified number of articles
            'published_after': start_date,
            'published_before': end_date
        }
        
        try:
            # Add user-agent to avoid potential blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            
            # Process the news data
            news_articles = []
            for article in data.get('data', []):
                # Get sentiment - handle potential None values
                sentiment = None
                if article.get('entities'):
                    for entity in article.get('entities', []):
                        if entity.get('sentiment_score') is not None:
                            sentiment = entity.get('sentiment_score')
                            break
                
                news_item = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'published_at': article.get('published_at', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'sentiment': sentiment,
                    'ticker': self.ticker  # Add ticker information
                }
                news_articles.append(news_item)
            
            # Cache the results
            self.news_cache[cache_key] = news_articles
            
            print(f"Retrieved {len(news_articles)} news articles for {self.ticker}")
            return news_articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {self.ticker}: {e}")
            # Create some dummy news data if API fails
            dummy_news = []
            for i in range(min(10, limit)):
                dummy_news.append({
                    'title': f"Dummy article {i+1} about {self.ticker}",
                    'description': f"This is a placeholder description for {self.ticker} when Marketaux API is unavailable.",
                    'published_at': (dt.datetime.now() - dt.timedelta(days=i)).isoformat(),
                    'url': 'https://example.com',
                    'source': 'Dummy Source',
                    'sentiment': np.random.uniform(-0.3, 0.3),  # Random sentiment between -0.3 and 0.3
                    'ticker': self.ticker
                })
            
            print(f"Using {len(dummy_news)} dummy news articles for {self.ticker}")
            self.news_cache[cache_key] = dummy_news
            return dummy_news
    
    def finetune_model_on_marketaux(self, days_back: int = 30):
        """
        Fine-tune the BERT model on sentiment data from Marketaux API.
        
        Args:
            days_back: Number of days to look back for news for training data
        """
        print("Collecting news data for fine-tuning...")
        
        # Get a larger set of news for training - request more articles
        news_articles = self.fetch_news(days_back, limit=50)
        
        # Filter articles that have sentiment scores from Marketaux
        training_data = []
        for article in news_articles:
            if article.get('sentiment') is not None:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text and len(text) > 10:
                    # Convert Marketaux sentiment score to class label
                    # Assuming Marketaux score is between -1 and 1
                    score = article.get('sentiment')
                    if score < -0.2:
                        label = 0  # Negative
                    elif score > 0.2:
                        label = 2  # Positive
                    else:
                        label = 1  # Neutral
                        
                    training_data.append((text, label))
        
        if len(training_data) < 3:
            print(f"Not enough labeled data for fine-tuning (found {len(training_data)}). Skipping fine-tuning.")
            self.model.eval()
            return
            
        print(f"Fine-tuning BERT model on {len(training_data)} news articles with Marketaux sentiment labels...")
        
        # Prepare datasets for fine-tuning
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
                self.labels = torch.tensor(labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
            
            def __len__(self):
                return len(self.labels)
        
        # Create training dataset
        texts, labels = zip(*training_data) if training_data else ([], [])
        dataset = SentimentDataset(list(texts), list(labels), self.tokenizer)
        
        # Set up training parameters
        batch_size = min(4, len(dataset))  # Small batch size for limited data
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer
        from torch.optim import AdamW
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        for epoch in range(self.finetune_epochs):
            total_loss = 0
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass and update
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.finetune_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Set model back to evaluation mode
        self.model.eval()
        print("Fine-tuning complete!")
        
        # Clear sentiment cache as the model has changed
        self.sentiment_cache = {}

    def analyze_sentiment_with_bert(self, texts: List[str]) -> float:
        """
        Analyze sentiment of text using BERT model.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not texts or self.model is None or self.tokenizer is None:
            return 0.0
        
        # Filter out empty or extremely short texts
        filtered_texts = [text for text in texts if text and len(text) > 10]
        
        if not filtered_texts:
            return 0.0
        
        # Check cache first (use concatenated texts as key with a hash to limit size)
        cache_key = str(hash("_".join(filtered_texts)))
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        sentiment_scores = []
        
        try:
            for text in filtered_texts:
                # Truncate very long texts to avoid memory issues
                if len(text) > 1024:
                    text = text[:1024]
                
                # Tokenize and prepare input
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # Get sentiment prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    # For FinBERT model (labels: negative, neutral, positive)
                    if probs.shape[0] == 3:
                        # Calculate sentiment score: Positive - Negative
                        sentiment_score = float(probs[2] - probs[0])
                    else:
                        # Fallback for other models - assume binary classification
                        sentiment_score = float(probs[1] * 2 - 1)
                    
                    sentiment_scores.append(sentiment_score)
            
            # Average sentiment score
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Cache the result
            self.sentiment_cache[cache_key] = avg_sentiment
            
            return avg_sentiment
            
        except Exception as e:
            print(f"Error in BERT sentiment analysis: {e}")
            return 0.0  # Return neutral sentiment in case of errors
    
    def get_combined_sentiment(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get combined sentiment from Marketaux API and BERT analysis.
        
        Args:
            days_back: Number of days to look back for news
            
        Returns:
            Dictionary with sentiment scores and analysis details
        """
        try:
            # Fetch news from Marketaux - limit to 10 articles
            news_articles = self.fetch_news(days_back, limit=10)
            
            if not news_articles:
                print("No news articles found. Using neutral sentiment.")
                return {
                    'marketaux_sentiment': 0,
                    'bert_sentiment': 0,
                    'combined_sentiment': 0,
                    'news_count': 0,
                    'details': []
                }
            
            # Extract text for BERT analysis
            texts = [f"{article.get('title', '')} {article.get('description', '')}" for article in news_articles]
            
            # Get Marketaux sentiment (if available in the API response)
            # Filter out None values to avoid TypeError
            marketaux_scores = []
            for article in news_articles:
                sentiment = article.get('sentiment')
                if sentiment is not None:  # Only include non-None values
                    marketaux_scores.append(float(sentiment))
            
            marketaux_sentiment = np.mean(marketaux_scores) if marketaux_scores else 0
            
            # Get BERT sentiment if model is available
            if self.model is not None:
                bert_sentiment = self.analyze_sentiment_with_bert(texts)
            else:
                # Use simple lexicon-based sentiment as fallback
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                try:
                    import nltk
                    nltk.download('vader_lexicon', quiet=True)
                    sid = SentimentIntensityAnalyzer()
                    scores = [sid.polarity_scores(text)['compound'] for text in texts]
                    bert_sentiment = np.mean(scores) if scores else 0
                except Exception as e:
                    print(f"Error using NLTK sentiment: {e}. Using neutral sentiment.")
                    bert_sentiment = 0
            
            # Combine sentiments (equal weighting, but can be adjusted)
            # Use BERT sentiment if Marketaux sentiment is not available
            if marketaux_scores:
                combined_sentiment = (marketaux_sentiment + bert_sentiment) / 2
            else:
                combined_sentiment = bert_sentiment
            
            # Prepare detailed results
            details = []
            for i, article in enumerate(news_articles):
                article_sentiment = None
                if self.model is not None and i < len(texts):
                    try:
                        article_sentiment = self.analyze_sentiment_with_bert([texts[i]])
                    except Exception as e:
                        print(f"Error analyzing sentiment for article: {e}")
                
                details.append({
                    'title': article.get('title', ''),
                    'published_at': article.get('published_at', ''),
                    'source': article.get('source', ''),
                    'marketaux_sentiment': article.get('sentiment'),
                    'bert_sentiment': article_sentiment
                })
            
            return {
                'marketaux_sentiment': marketaux_sentiment,
                'bert_sentiment': bert_sentiment,
                'combined_sentiment': combined_sentiment,
                'news_count': len(news_articles),
                'details': details
            }
        except Exception as e:
            print(f"Error in get_combined_sentiment: {e}")
            # Return neutral sentiment in case of errors
            return {
                'marketaux_sentiment': 0,
                'bert_sentiment': 0, 
                'combined_sentiment': 0,
                'news_count': 0,
                'details': [],
                'error': str(e)
            }
    
    def adjust_volatility(self, base_volatility: float, sentiment_score: float, 
                         sensitivity: float = 0.2) -> float:
        """
        Adjust volatility based on sentiment.
        
        Args:
            base_volatility: Base volatility from historical data or implied vol
            sentiment_score: Sentiment score (-1 to 1)
            sensitivity: Parameter controlling sensitivity to sentiment
            
        Returns:
            Adjusted volatility
        """
        # Negative sentiment increases volatility, positive sentiment decreases it
        adjustment = 1.0 - (sentiment_score * sensitivity)
        adjusted_volatility = base_volatility * adjustment
        
        # Ensure volatility stays positive
        return max(0.001, adjusted_volatility)
    
    def adjust_sabr_params(self, 
                          alpha: float, 
                          beta: float, 
                          rho: float, 
                          volvol: float, 
                          sentiment_score: float, 
                          sensitivity: float = 0.15) -> Tuple[float, float, float, float]:
        """
        Adjust SABR parameters based on sentiment.
        
        Args:
            alpha: SABR alpha (ATM volatility)
            beta: SABR beta (controls skew)
            rho: SABR rho (correlation between price and vol)
            volvol: SABR volvol (volatility of volatility)
            sentiment_score: Sentiment score (-1 to 1)
            sensitivity: Sensitivity of adjustment to sentiment
            
        Returns:
            Tuple of adjusted SABR parameters
        """
        # Adjust alpha (ATM volatility) - lower for positive sentiment
        adjusted_alpha = alpha * (1.0 - sentiment_score * sensitivity)
        
        # Adjust rho (correlation) - more negative for negative sentiment
        adjusted_rho = rho - sentiment_score * sensitivity * 0.5
        adjusted_rho = max(-0.99, min(0.99, adjusted_rho))  # Keep in valid range
        
        # Adjust volvol - higher for negative sentiment
        adjusted_volvol = volvol * (1.0 - sentiment_score * sensitivity * 0.5)
        
        return adjusted_alpha, beta, adjusted_rho, adjusted_volvol
    
    def black_scholes_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           vol: float, 
                           option_type: str = 'call') -> float:
        """
        Price an option using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Calculate Black-Scholes price using pysabr
        if option_type.lower() == 'call':
            return black.lognormal_call(K, S, T, vol, r, 'call')
        else:
            return black.lognormal_call(K, S, T, vol, r, 'put')
    
    def monte_carlo_price(self, 
                         S: float, 
                         K: float, 
                         T: float, 
                         r: float, 
                         vol: float, 
                         option_type: str = 'call',
                         n_simulations: int = 10000,
                         n_steps: int = 252,
                         sentiment_score: float = 0) -> Dict[str, Any]:
        """
        Price an option using Monte Carlo simulation with sentiment-adjusted drift.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            n_simulations: Number of simulation paths
            n_steps: Number of time steps
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Dictionary with option price and simulation details
        """
        dt = T / n_steps
        
        # Adjust drift based on sentiment (higher sentiment = higher expected return)
        sentiment_drift_adj = 0.02 * sentiment_score  # Parameter can be tuned
        adjusted_drift = r + sentiment_drift_adj
        
        # Initialize price paths
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S
        
        # Generate price paths
        for t in range(1, n_steps + 1):
            # Generate random shocks
            Z = np.random.standard_normal(n_simulations)
            
            # Update stock prices
            paths[:, t] = paths[:, t-1] * np.exp((adjusted_drift - 0.5 * vol**2) * dt + 
                                                vol * np.sqrt(dt) * Z)
        
        # Calculate option payoffs at maturity
        if option_type.lower() == 'call':
            payoffs = np.maximum(paths[:, -1] - K, 0)
        else:  # put option
            payoffs = np.maximum(K - paths[:, -1], 0)
        
        # Discount payoffs to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            'price': option_price,
            'std_error': np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations),
            'paths': paths,
            'payoffs': payoffs,
            'adjusted_drift': adjusted_drift
        }
    
    def price_option(self, 
                    S: float, 
                    K: float, 
                    T: float, 
                    r: float, 
                    vol: float, 
                    option_type: str = 'call',
                    method: str = 'black_scholes',
                    days_back: int = 7,
                    sentiment_sensitivity: float = 0.2) -> Dict[str, Any]:
        """
        Price an option using the selected method with sentiment adjustment.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            vol: Base volatility
            option_type: 'call' or 'put'
            method: 'black_scholes' or 'monte_carlo'
            days_back: Number of days to look back for news
            sentiment_sensitivity: Sensitivity to sentiment in volatility adjustment
            
        Returns:
            Dictionary with pricing results and sentiment analysis
        """
        # Get sentiment for the ticker
        sentiment_data = self.get_combined_sentiment(days_back)
        sentiment_score = sentiment_data['combined_sentiment']
        
        # Adjust volatility based on sentiment
        adjusted_vol = self.adjust_volatility(vol, sentiment_score, sentiment_sensitivity)
        
        # Price the option using the selected method
        if method.lower() == 'black_scholes':
            price = self.black_scholes_price(S, K, T, r, adjusted_vol, option_type)
            # Also calculate the price without sentiment adjustment for comparison
            baseline_price = self.black_scholes_price(S, K, T, r, vol, option_type)
        elif method.lower() == 'monte_carlo':
            mc_result = self.monte_carlo_price(S, K, T, r, adjusted_vol, option_type, 
                                              sentiment_score=sentiment_score)
            price = mc_result['price']
            # Also calculate the price without sentiment adjustment
            baseline_mc = self.monte_carlo_price(S, K, T, r, vol, option_type)
            baseline_price = baseline_mc['price']
        else:
            raise ValueError("Method must be 'black_scholes' or 'monte_carlo'")
        
        return {
            'price': price,
            'baseline_price': baseline_price,
            'price_difference': price - baseline_price,
            'price_difference_percent': (price - baseline_price) / baseline_price * 100 if baseline_price else 0,
            'base_volatility': vol,
            'adjusted_volatility': adjusted_vol,
            'sentiment_score': sentiment_score,
            'sentiment_data': sentiment_data,
            'option_params': {
                'S': S,
                'K': K,
                'T': T,
                'r': r,
                'option_type': option_type
            }
        }
    
    def calculate_implied_volatility(self, 
                                    S: float, 
                                    K: float, 
                                    T: float, 
                                    r: float, 
                                    option_price: float, 
                                    option_type: str = 'call') -> float:
        """
        Calculate implied volatility using bisection method.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            option_price: Market price of the option
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        # Set up bisection method
        max_iterations = 100
        precision = 1.0e-5
        
        # Initial volatility bounds
        vol_low = 0.001
        vol_high = 5.0
        
        # Bisection algorithm
        for i in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            
            # Calculate option price at mid volatility
            if option_type.lower() == 'call':
                price_mid = black.lognormal_call(K, S, T, vol_mid, r, 'call')
            else:
                price_mid = black.lognormal_call(K, S, T, vol_mid, r, 'put')
            
            # Check if precision is achieved
            if abs(price_mid - option_price) < precision:
                return vol_mid
            
            # Update bounds
            if price_mid > option_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
                
        # Return best estimate if max iterations reached
        return (vol_low + vol_high) / 2

    def plot_sentiment_impact(self, 
                             S: float, 
                             K: float, 
                             T: float, 
                             r: float, 
                             vol: float, 
                             option_type: str = 'call'):
        """
        Plot the impact of different sentiment scores on option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            vol: Base volatility
            option_type: 'call' or 'put'
        """
        sentiment_range = np.linspace(-1, 1, 21)
        prices = []
        vols = []
        
        for sentiment in sentiment_range:
            adjusted_vol = self.adjust_volatility(vol, sentiment)
            vols.append(adjusted_vol)
            
            if option_type.lower() == 'call':
                price = black.lognormal_call(K, S, T, adjusted_vol, r, 'call')
            else:
                price = black.lognormal_call(K, S, T, adjusted_vol, r, 'put')
                
            prices.append(price)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot price vs sentiment
        ax1.plot(sentiment_range, prices, 'b-', linewidth=2)
        ax1.set_title(f'Impact of Sentiment on {option_type.capitalize()} Option Price')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Option Price')
        ax1.grid(True)
        
        # Add baseline price
        if option_type.lower() == 'call':
            baseline_price = black.lognormal_call(K, S, T, vol, r, 'call')
        else:
            baseline_price = black.lognormal_call(K, S, T, vol, r, 'put')
            
        ax1.axhline(y=baseline_price, color='r', linestyle='--', 
                   label=f'Baseline Price: {baseline_price:.2f}')
        ax1.legend()
        
        # Plot adjusted volatility vs sentiment
        ax2.plot(sentiment_range, vols, 'g-', linewidth=2)
        ax2.set_title('Impact of Sentiment on Volatility')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Adjusted Volatility')
        ax2.axhline(y=vol, color='r', linestyle='--', 
                   label=f'Base Volatility: {vol:.2f}')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('sentiment_impact.png')
        print("Plot saved to sentiment_impact.png")
    
    def plot_volatility_surface(self, 
                               S: float, 
                               strikes: List[float], 
                               maturities: List[float], 
                               r: float, 
                               base_vol: float,
                               sentiment_score: float):
        """
        Plot volatility surface with and without sentiment adjustment.
        
        Args:
            S: Current stock price
            strikes: List of strike prices
            maturities: List of maturities (in years)
            r: Risk-free rate
            base_vol: Base volatility
            sentiment_score: Sentiment score (-1 to 1)
        """
        # Create meshgrid for surface
        K, T = np.meshgrid(strikes, maturities)
        
        # Initialize volatility surfaces
        vol_surface_base = np.zeros_like(K, dtype=float)
        vol_surface_adjusted = np.zeros_like(K, dtype=float)
        
        # Simple volatility skew model (can be replaced with SABR or SVI)
        for i, t in enumerate(maturities):
            for j, k in enumerate(strikes):
                moneyness = k / S
                
                # Base volatility with skew
                skew_factor = 0.1 * (1 - moneyness) * np.sqrt(t)
                vol_surface_base[i, j] = base_vol + skew_factor
                
                # Sentiment-adjusted volatility
                adjusted_vol = self.adjust_volatility(base_vol, sentiment_score)
                vol_surface_adjusted[i, j] = adjusted_vol + skew_factor
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Base volatility surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(K, T, vol_surface_base, cmap='viridis', 
                                alpha=0.8, linewidth=0, antialiased=True)
        ax1.set_title('Base Volatility Surface')
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity (years)')
        ax1.set_zlabel('Implied Volatility')
        
        # Sentiment-adjusted volatility surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(K, T, vol_surface_adjusted, cmap='plasma', 
                                alpha=0.8, linewidth=0, antialiased=True)
        ax2.set_title(f'Sentiment-Adjusted Volatility Surface\nSentiment: {sentiment_score:.2f}')
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity (years)')
        ax2.set_zlabel('Implied Volatility')
        
        plt.tight_layout()
        plt.savefig('volatility_surface.png')
        print("Plot saved to volatility_surface.png")
    
# Example usage
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Sentiment-Enhanced Options Pricing')
    parser.add_argument('--ticker', type=str, default="AAPL", help='Stock ticker symbol')
    parser.add_argument('--api_key', type=str, required=True, help='Marketaux API key')
    parser.add_argument('--stock_price', type=float, default=150.0, help='Current stock price')
    parser.add_argument('--strike_price', type=float, default=155.0, help='Option strike price')
    parser.add_argument('--time_to_maturity', type=float, default=0.25, help='Time to maturity in years')
    parser.add_argument('--risk_free_rate', type=float, default=0.05, help='Risk-free interest rate')
    parser.add_argument('--volatility', type=float, default=0.30, help='Base volatility')
    parser.add_argument('--option_type', type=str, default='call', choices=['call', 'put'], help='Option type')
    parser.add_argument('--method', type=str, default='black_scholes', choices=['black_scholes', 'monte_carlo'], help='Pricing method')
    parser.add_argument('--days_back', type=int, default=7, help='Number of days to look back for news')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    parser.add_argument('--use_finbert', action='store_true', help='Use FinBERT instead of base BERT')
    parser.add_argument('--no_finetune', action='store_true', help='Disable fine-tuning on Marketaux data')
    parser.add_argument('--finetune_epochs', type=int, default=3, help='Number of epochs for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory for saving output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Initialize model with API key
        print(f"Initializing model for {args.ticker}...")
        model = SentimentEnhancedOptionsPricing(
            ticker=args.ticker,
            marketaux_api_key=args.api_key,
            use_pretrained_sentiment_model=args.use_finbert,
            finetune_on_marketaux=not args.no_finetune,
            finetune_epochs=args.finetune_epochs
        )
        
        print(f"\nPricing {args.option_type} option for {args.ticker}...")
        # Price the option
        result = model.price_option(
            S=args.stock_price,
            K=args.strike_price,
            T=args.time_to_maturity,
            r=args.risk_free_rate,
            vol=args.volatility,
            option_type=args.option_type,
            method=args.method,
            days_back=args.days_back
        )
        
        # Print results
        print("\n===== SENTIMENT ANALYSIS =====")
        print(f"Number of news articles: {result['sentiment_data']['news_count']}")
        print(f"Marketaux API Sentiment: {result['sentiment_data']['marketaux_sentiment']:.4f}")
        print(f"BERT Model Sentiment (fine-tuned on Marketaux data): {result['sentiment_data']['bert_sentiment']:.4f}")
        print(f"Combined Sentiment Score: {result['sentiment_score']:.4f}")
        
        print("\n===== OPTION PRICING =====")
        print(f"Base Volatility: {result['base_volatility']:.4f}")
        print(f"Sentiment-Adjusted Volatility: {result['adjusted_volatility']:.4f}")
        print(f"Base Option Price: ${result['baseline_price']:.2f}")
        print(f"Sentiment-Adjusted Price: ${result['price']:.2f}")
        print(f"Price Difference: ${result['price_difference']:.2f} ({result['price_difference_percent']:.2f}%)")
        
        # Sample of news details
        print("\n===== SAMPLE NEWS =====")
        for i, detail in enumerate(result['sentiment_data']['details'][:3]):  # Show first 3 news items
            print(f"{i+1}. {detail['title']}")
            print(f"   Published: {detail['published_at']}")
            print(f"   Source: {detail['source']}")
            print(f"   Marketaux Sentiment: {detail['marketaux_sentiment']}")
            
            bert_sentiment = detail['bert_sentiment']
            if bert_sentiment is not None:
                print(f"   BERT Sentiment: {bert_sentiment:.4f}")
            else:
                print(f"   BERT Sentiment: N/A")
            print()
        
        # Plot sentiment impact
        if not args.no_plot:
            print("\nGenerating sentiment impact visualization...")
            plot_path = os.path.join(args.output_dir, f"{args.ticker}_{args.option_type}_sentiment_impact.png")
            model.plot_sentiment_impact(
                S=args.stock_price,
                K=args.strike_price,
                T=args.time_to_maturity,
                r=args.risk_free_rate,
                vol=args.volatility,
                option_type=args.option_type,
            )
            
            # Generate volatility surface plot
            print("\nGenerating volatility surface visualization...")
            strikes = np.linspace(args.stock_price * 0.8, args.stock_price * 1.2, 10)
            maturities = np.array([0.1, 0.25, 0.5, 1.0])
            vol_surface_path = os.path.join(args.output_dir, f"{args.ticker}_vol_surface.png")
            model.plot_volatility_surface(
                S=args.stock_price, 
                strikes=strikes,
                maturities=maturities,
                r=args.risk_free_rate,
                base_vol=args.volatility,
                sentiment_score=result['sentiment_score'],
            )
            
        # Save results to file
        import json
        
        # Convert complex objects to serializable format
        def serialize_result(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            else:
                return str(obj)
        
        # Remove large arrays from result to avoid huge files
        result_for_saving = {k: v for k, v in result.items() if k not in ['paths', 'payoffs']}
        
        with open(os.path.join(args.output_dir, f"{args.ticker}_{args.option_type}_result.json"), 'w') as f:
            json.dump(result_for_saving, f, default=serialize_result, indent=2)
        
        print(f"\nResults saved to {args.output_dir}")
            
    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        traceback.print_exc()