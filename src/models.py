"""
SQLAlchemy models for Harvester II database.
Provides ORM models and schema validation for portfolio and data caching.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import MetaData, inspect
from sqlalchemy.sql import func
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

# Data Manager Models (for caching)
class PriceCache(Base):
    """Price data cache table."""
    __tablename__ = 'price_cache'

    symbol = Column(String(20), primary_key=True)
    date = Column(String(10), primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    timestamp = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_price_symbol_date', 'symbol', 'date'),
        Index('idx_price_timestamp', 'timestamp'),
    )


class TrendsCache(Base):
    """Google Trends data cache table."""
    __tablename__ = 'trends_cache'

    keyword = Column(String(100), primary_key=True)
    date = Column(String(10), primary_key=True)
    value = Column(Integer)
    timestamp = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_trends_keyword_date', 'keyword', 'date'),
        Index('idx_trends_timestamp', 'timestamp'),
    )


class MacroCache(Base):
    """Macro indicators cache table."""
    __tablename__ = 'macro_cache'

    indicator = Column(String(20), primary_key=True)
    date = Column(String(10), primary_key=True)
    value = Column(Float)
    timestamp = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_macro_indicator_date', 'indicator', 'date'),
        Index('idx_macro_timestamp', 'timestamp'),
    )


# Portfolio Manager Models
class Position(Base):
    """Portfolio positions table."""
    __tablename__ = 'positions'

    symbol = Column(String(20), primary_key=True)
    shares = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    stop_loss = Column(Float)
    profit_target = Column(Float)
    atr = Column(Float)
    g_score = Column(Float)
    position_value = Column(Float)
    risk_amount = Column(Float)
    status = Column(String(20), default='open')  # 'open', 'closed', 'pending'

    __table_args__ = (
        Index('idx_position_status', 'status'),
        Index('idx_position_entry_time', 'entry_time'),
    )


class Order(Base):
    """Trading orders table."""
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    shares = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    order_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), default='pending')  # 'pending', 'filled', 'cancelled'
    order_type = Column(String(20), default='market')  # 'market', 'limit', etc.
    fill_price = Column(Float)
    fill_time = Column(DateTime)

    __table_args__ = (
        Index('idx_order_symbol', 'symbol'),
        Index('idx_order_status', 'status'),
        Index('idx_order_time', 'order_time'),
    )


class TradeHistory(Base):
    """Trade history table."""
    __tablename__ = 'trade_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    shares = Column(Integer, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    pnl = Column(Float, nullable=False)
    pnl_percentage = Column(Float, nullable=False)
    exit_reason = Column(String(100))
    duration_days = Column(Float)
    side = Column(String(10), nullable=False)

    __table_args__ = (
        Index('idx_trade_symbol', 'symbol'),
        Index('idx_trade_entry_time', 'entry_time'),
        Index('idx_trade_exit_time', 'exit_time'),
        Index('idx_trade_pnl', 'pnl'),
    )


class DatabaseManager:
    """Manages SQLAlchemy database connections and schema validation."""

    def __init__(self, db_path: str, echo: bool = False, encrypted: bool = False):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
            encrypted: Whether to use encrypted database (SQLCipher)
        """
        self.db_path = db_path
        self.encrypted = encrypted

        if encrypted:
            # Use SQLCipher for encrypted database
            try:
                from sqlalchemy.engine import make_url
                from sqlcipher3 import dbapi2 as sqlcipher

                # Get encryption key from environment variable
                key = os.getenv('DATABASE_ENCRYPTION_KEY', '')
                if not key:
                    logger.warning("DATABASE_ENCRYPTION_KEY not set, falling back to standard SQLite")
                    self.engine = create_engine(f'sqlite:///{db_path}', echo=echo)
                else:
                    # Create SQLCipher URL
                    url = make_url(f'sqlite:///{db_path}')
                    url.query = {'key': key}

                    self.engine = create_engine(url, echo=echo, module=sqlcipher)
                    logger.info("Using encrypted database with SQLCipher")

            except ImportError:
                logger.warning("sqlcipher3 package not available, falling back to standard SQLite")
                self.engine = create_engine(f'sqlite:///{db_path}', echo=echo)
        else:
            self.engine = create_engine(f'sqlite:///{db_path}', echo=echo)

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database tables created successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def validate_schema(self) -> bool:
        """
        Validate database schema using SQLAlchemy metadata reflection.

        Returns:
            True if schema is valid
        """
        try:
            # Reflect existing database schema
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            inspector = inspect(self.engine)

            # Check required tables exist
            required_tables = [
                'price_cache', 'trends_cache', 'macro_cache',
                'positions', 'orders', 'trade_history'
            ]

            existing_tables = inspector.get_table_names()

            for table_name in required_tables:
                if table_name not in existing_tables:
                    logger.error(f"Required table '{table_name}' not found in database")
                    return False

            # Validate table structures
            if not self._validate_price_cache_schema(inspector):
                return False
            if not self._validate_trends_cache_schema(inspector):
                return False
            if not self._validate_macro_cache_schema(inspector):
                return False
            if not self._validate_positions_schema(inspector):
                return False
            if not self._validate_orders_schema(inspector):
                return False
            if not self._validate_trade_history_schema(inspector):
                return False

            logger.info("Database schema validation passed")
            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def _validate_price_cache_schema(self, inspector) -> bool:
        """Validate price_cache table schema."""
        try:
            columns = inspector.get_columns('price_cache')
            column_names = [col['name'] for col in columns]

            required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"price_cache missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating price_cache schema: {e}")
            return False

    def _validate_trends_cache_schema(self, inspector) -> bool:
        """Validate trends_cache table schema."""
        try:
            columns = inspector.get_columns('trends_cache')
            column_names = [col['name'] for col in columns]

            required_columns = ['keyword', 'date', 'value', 'timestamp']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"trends_cache missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating trends_cache schema: {e}")
            return False

    def _validate_macro_cache_schema(self, inspector) -> bool:
        """Validate macro_cache table schema."""
        try:
            columns = inspector.get_columns('macro_cache')
            column_names = [col['name'] for col in columns]

            required_columns = ['indicator', 'date', 'value', 'timestamp']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"macro_cache missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating macro_cache schema: {e}")
            return False

    def _validate_positions_schema(self, inspector) -> bool:
        """Validate positions table schema."""
        try:
            columns = inspector.get_columns('positions')
            column_names = [col['name'] for col in columns]

            required_columns = ['symbol', 'shares', 'entry_price', 'entry_time', 'side', 'status']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"positions missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating positions schema: {e}")
            return False

    def _validate_orders_schema(self, inspector) -> bool:
        """Validate orders table schema."""
        try:
            columns = inspector.get_columns('orders')
            column_names = [col['name'] for col in columns]

            required_columns = ['id', 'symbol', 'side', 'shares', 'price', 'order_time', 'status']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"orders missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating orders schema: {e}")
            return False

    def _validate_trade_history_schema(self, inspector) -> bool:
        """Validate trade_history table schema."""
        try:
            columns = inspector.get_columns('trade_history')
            column_names = [col['name'] for col in columns]

            required_columns = ['id', 'symbol', 'entry_price', 'exit_price', 'shares', 'entry_time', 'exit_time', 'pnl', 'pnl_percentage', 'side']
            for col in required_columns:
                if col not in column_names:
                    logger.error(f"trade_history missing required column: {col}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating trade_history schema: {e}")
            return False

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")


# Global database managers
_data_db_manager: Optional[DatabaseManager] = None
_portfolio_db_manager: Optional[DatabaseManager] = None


def get_data_db_manager() -> DatabaseManager:
    """Get the global data database manager instance."""
    global _data_db_manager
    if _data_db_manager is None:
        from config import get_config
        config = get_config()

        db_path = "data/harvester_ii.db"
        Path(db_path).parent.mkdir(exist_ok=True)

        encrypted = config.get('database.encrypted', False)
        _data_db_manager = DatabaseManager(db_path, encrypted=encrypted)
        _data_db_manager.create_tables()
        if not _data_db_manager.validate_schema():
            raise RuntimeError("Data database schema validation failed")
    return _data_db_manager


def get_portfolio_db_manager() -> DatabaseManager:
    """Get the global portfolio database manager instance."""
    global _portfolio_db_manager
    if _portfolio_db_manager is None:
        from config import get_config
        config = get_config()

        db_path = "data/portfolio.db"
        Path(db_path).parent.mkdir(exist_ok=True)

        encrypted = config.get('database.encrypted', False)
        _portfolio_db_manager = DatabaseManager(db_path, encrypted=encrypted)
        _portfolio_db_manager.create_tables()
        if not _portfolio_db_manager.validate_schema():
            raise RuntimeError("Portfolio database schema validation failed")
    return _portfolio_db_manager
