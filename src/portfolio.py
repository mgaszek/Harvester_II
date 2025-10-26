"""
Portfolio management module for Harvester II trading system.
Handles position tracking, order execution, and portfolio monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Dependencies are now injected via constructor
from models import get_portfolio_db_manager, Position, Order, TradeHistory


class PortfolioManager:
    """Manages portfolio positions, orders, and execution."""
    
    def __init__(self, config, risk_manager, data_manager, signal_calculator):
        """Initialize portfolio manager with injected dependencies."""
        self.config = config
        self.risk_manager = risk_manager
        self.data_manager = data_manager
        self.signal_calculator = signal_calculator
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for portfolio tracking
        self._init_portfolio_db()
        
        # Portfolio state
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.trade_history: List[Dict] = {}
        
        # Load existing positions from database
        
        self.logger.info("Portfolio Manager initialized")
    
    def _init_portfolio_db(self) -> None:
        """Initialize portfolio database with SQLAlchemy."""
        try:
            self.db_manager = get_portfolio_db_manager()
            self.logger.info("Portfolio database initialized with SQLAlchemy")
        except Exception as e:
            self.logger.error(f"Failed to initialize portfolio database: {e}")
            self.db_manager = None
    
    def _load_positions(self) -> None:
        """Load existing open positions from database."""
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            positions = session.query(Position).filter_by(status='open').all()

            for pos in positions:
                self.positions[pos.symbol] = {
                    'symbol': pos.symbol,
                    'shares': pos.shares,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time,
                    'side': pos.side,
                    'stop_loss': pos.stop_loss,
                    'profit_target': pos.profit_target,
                    'atr': pos.atr,
                    'g_score': pos.g_score,
                    'position_value': pos.position_value,
                    'risk_amount': pos.risk_amount,
                    'status': pos.status
                }

            self.logger.info(f"Loaded {len(self.positions)} open positions from database")

        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
        finally:
            session.close()
    
    def execute_entry_order(self, signal: Dict[str, Any], 
                           current_price: float) -> bool:
        """
        Execute an entry order based on signal.
        
        Args:
            signal: Entry signal from signal calculator
            current_price: Current market price
            
        Returns:
            True if order executed successfully
        """
        try:
            symbol = signal['symbol']
            side = signal['side']
            
            # Check if position already exists
            if symbol in self.positions:
                self.logger.warning(f"Position already exists for {symbol}")
                return False
            
            # Get ATR for position sizing
            price_data = self.data_manager.get_price_data(symbol, period="1mo")
            if price_data.empty:
                self.logger.error(f"No price data for {symbol}")
                return False
            
            # Calculate technical indicators
            price_data = self.data_manager.calculate_technical_indicators(price_data)
            atr = price_data['ATR'].iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                self.logger.error(f"Invalid ATR for {symbol}: {atr}")
                return False
            
            # Get G-Score for risk adjustment
            g_score = self.signal_calculator.calculate_g_score()
            
            # Calculate position size
            position_info = self.risk_manager.calculate_position_size(
                symbol, current_price, atr, g_score
            )
            
            if not position_info:
                self.logger.error(f"Failed to calculate position size for {symbol}")
                return False
            
            # Add position to risk manager
            if not self.risk_manager.add_position(position_info):
                self.logger.error(f"Risk manager rejected position for {symbol}")
                return False
            
            # Create position record
            position = {
                'symbol': symbol,
                'shares': position_info['shares'],
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'side': side,
                'stop_loss': position_info['stop_loss_price'],
                'profit_target': position_info['profit_target_price'],
                'atr': atr,
                'g_score': g_score,
                'position_value': position_info['position_value'],
                'risk_amount': position_info['risk_amount'],
                'status': 'open'
            }
            
            # Store in portfolio
            self.positions[symbol] = position
            
            # Save to database
            self._save_position(position)
            
            # Create order record
            order = {
                'symbol': symbol,
                'side': side,
                'shares': position_info['shares'],
                'price': current_price,
                'order_time': datetime.now(),
                'status': 'filled',
                'order_type': 'market',
                'fill_price': current_price,
                'fill_time': datetime.now()
            }
            
            self.orders.append(order)
            self._save_order(order)
            
            self.logger.info(f"Executed {side} order for {symbol}: "
                           f"{position_info['shares']} shares at ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute entry order for {signal['symbol']}: {e}")
            return False
    
    def execute_exit_order(self, symbol: str, exit_price: float, 
                          exit_reason: str) -> bool:
        """
        Execute an exit order for a position.
        
        Args:
            symbol: Asset symbol
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            True if order executed successfully
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Remove from risk manager
            trade_record = self.risk_manager.remove_position(symbol, exit_price, exit_reason)
            
            if not trade_record:
                self.logger.error(f"Failed to remove position from risk manager: {symbol}")
                return False
            
            # Update position status
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['exit_reason'] = exit_reason
            
            # Save trade to history
            self.trade_history.append(trade_record)
            self._save_trade_history(trade_record)
            
            # Update position in database
            self._update_position_status(symbol, 'closed')
            
            # Remove from active positions
            del self.positions[symbol]
            
            # Create exit order record
            order = {
                'symbol': symbol,
                'side': 'SELL' if position['side'] == 'BUY' else 'BUY',
                'shares': position['shares'],
                'price': exit_price,
                'order_time': datetime.now(),
                'status': 'filled',
                'order_type': 'market',
                'fill_price': exit_price,
                'fill_time': datetime.now()
            }
            
            self.orders.append(order)
            self._save_order(order)
            
            self.logger.info(f"Executed exit order for {symbol}: "
                           f"{position['shares']} shares at ${exit_price:.2f} - {exit_reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute exit order for {symbol}: {e}")
            return False
    
    def check_exit_signals(self) -> List[Dict[str, Any]]:
        """
        Check all open positions for exit signals.
        
        Returns:
            List of positions that should be exited
        """
        exits = []
        
        try:
            # Get current prices for all positions
            symbols = list(self.positions.keys())
            if not symbols:
                return exits
            
            current_prices = {}
            for symbol in symbols:
                price_data = self.data_manager.get_price_data(symbol, period="5d")
                if not price_data.empty:
                    current_prices[symbol] = price_data['Close'].iloc[-1]
            
            # Check exit signals using risk manager
            exit_signals = self.risk_manager.check_exit_signals(current_prices)
            
            for signal in exit_signals:
                symbol = signal['symbol']
                exit_price = signal['exit_price']
                exit_reason = signal['exit_reason']
                
                exits.append({
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'position': self.positions[symbol]
                })
            
            return exits
            
        except Exception as e:
            self.logger.error(f"Failed to check exit signals: {e}")
            return []
    
    def process_exit_signals(self) -> int:
        """
        Process all exit signals and execute orders.
        
        Returns:
            Number of positions closed
        """
        exits = self.check_exit_signals()
        closed_count = 0
        
        for exit_signal in exits:
            symbol = exit_signal['symbol']
            exit_price = exit_signal['exit_price']
            exit_reason = exit_signal['exit_reason']
            
            if self.execute_exit_order(symbol, exit_price, exit_reason):
                closed_count += 1
        
        if closed_count > 0:
            self.logger.info(f"Processed {closed_count} exit signals")
        
        return closed_count
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio statistics
        """
        try:
            # Get risk manager summary
            risk_summary = self.risk_manager.get_portfolio_summary()
            
            # Calculate portfolio-specific metrics
            total_positions = len(self.positions)
            total_orders = len(self.orders)
            total_trades = len(self.trade_history)
            
            # Calculate unrealized P&L (simplified)
            unrealized_pnl = 0.0
            for symbol, position in self.positions.items():
                # This would need current prices - placeholder for now
                unrealized_pnl += 0.0
            
            # Calculate realized P&L
            realized_pnl = sum(trade['pnl'] for trade in self.trade_history)
            
            # Calculate win rate
            winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate average trade duration
            avg_duration = 0.0
            if self.trade_history:
                durations = [trade['duration'].total_seconds() / 86400 for trade in self.trade_history]
                avg_duration = np.mean(durations)
            
            summary = {
                **risk_summary,
                'portfolio_metrics': {
                    'total_positions': total_positions,
                    'total_orders': total_orders,
                    'total_trades': total_trades,
                    'unrealized_pnl': unrealized_pnl,
                    'realized_pnl': realized_pnl,
                    'win_rate': win_rate,
                    'avg_trade_duration_days': avg_duration
                },
                'positions': list(self.positions.keys()),
                'last_update': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    def _save_position(self, position: Dict[str, Any]) -> None:
        """Save position to database using SQLAlchemy."""
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            # Check if position exists
            existing = session.query(Position).filter_by(symbol=position['symbol']).first()

            if existing:
                # Update existing position
                existing.shares = position['shares']
                existing.entry_price = position['entry_price']
                existing.entry_time = position['entry_time']
                existing.side = position['side']
                existing.stop_loss = position['stop_loss']
                existing.profit_target = position['profit_target']
                existing.atr = position['atr']
                existing.g_score = position['g_score']
                existing.position_value = position['position_value']
                existing.risk_amount = position['risk_amount']
                existing.status = position['status']
            else:
                # Create new position
                db_position = Position(
                    symbol=position['symbol'],
                    shares=position['shares'],
                    entry_price=position['entry_price'],
                    entry_time=position['entry_time'],
                    side=position['side'],
                    stop_loss=position['stop_loss'],
                    profit_target=position['profit_target'],
                    atr=position['atr'],
                    g_score=position['g_score'],
                    position_value=position['position_value'],
                    risk_amount=position['risk_amount'],
                    status=position['status']
                )
                session.add(db_position)

            session.commit()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save position: {e}")
        finally:
            session.close()
    
    def _save_order(self, order: Dict[str, Any]) -> None:
        """Save order to database using SQLAlchemy."""
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            db_order = Order(
                symbol=order['symbol'],
                side=order['side'],
                shares=order['shares'],
                price=order['price'],
                order_time=order['order_time'],
                status=order['status'],
                order_type=order.get('order_type', 'market'),
                fill_price=order.get('fill_price'),
                fill_time=order.get('fill_time')
            )
            session.add(db_order)
            session.commit()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save order: {e}")
        finally:
            session.close()
    
    def _save_trade_history(self, trade: Dict[str, Any]) -> None:
        """Save trade to history database using SQLAlchemy."""
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            db_trade = TradeHistory(
                symbol=trade['symbol'],
                entry_price=trade['entry_price'],
                exit_price=trade['exit_price'],
                shares=trade['shares'],
                entry_time=trade['entry_time'],
                exit_time=trade['exit_time'],
                pnl=trade['pnl'],
                pnl_percentage=trade['pnl_percentage'],
                exit_reason=trade.get('exit_reason'),
                duration_days=trade['duration'].total_seconds() / 86400 if 'duration' in trade else None,
                side=trade.get('side', 'long')
            )
            session.add(db_trade)
            session.commit()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save trade history: {e}")
        finally:
            session.close()
    
    def _update_position_status(self, symbol: str, status: str) -> None:
        """Update position status in database using SQLAlchemy."""
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            position = session.query(Position).filter_by(symbol=symbol).first()
            if position:
                position.status = status
                session.commit()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to update position status: {e}")
        finally:
            session.close()
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def close(self) -> None:
        """Close database connection."""
        if self.db_manager:
            self.db_manager.close()
            self.logger.info("Portfolio database connection closed")


# PortfolioManager is now created via dependency injection in di.py