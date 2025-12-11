"""
Order Management Module

This module handles order tracking, status lookup, and order-related queries.
It provides sample order data for demonstration purposes.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class OrderManager:
    """
    Manages order data and provides order lookup functionality.
    """
    
    def __init__(self, orders_file: str = None):
        """
        Initialize the OrderManager with order data.
        
        Args:
            orders_file: Path to the JSON file containing order data
        """
        if orders_file is None:
            orders_file = Path(__file__).parent / "data" / "orders" / "sample_orders.json"
        
        self.orders_file = Path(orders_file)
        self.orders = {}
        self.carriers = {}
        self._load_orders()
    
    def _load_orders(self):
        """Load orders from JSON file."""
        if self.orders_file.exists():
            with open(self.orders_file, 'r') as f:
                data = json.load(f)
                # Index orders by order_id for quick lookup
                self.orders = {order['order_id']: order for order in data.get('orders', [])}
                self.carriers = data.get('tracking_carriers', {})
            print(f"✓ Loaded {len(self.orders)} orders from database")
        else:
            print(f"Warning: Orders file not found at {self.orders_file}")
            self.orders = {}
            self.carriers = {}
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order details by order ID.
        
        Args:
            order_id: The order ID (e.g., "TM-2024-001234")
            
        Returns:
            Order dictionary or None if not found
        """
        # Normalize order ID (uppercase, handle partial matches)
        order_id = order_id.upper().strip()
        
        # Direct lookup
        if order_id in self.orders:
            return self.orders[order_id]
        
        # Try with TM- prefix if not provided
        if not order_id.startswith('TM-'):
            prefixed_id = f"TM-{order_id}"
            if prefixed_id in self.orders:
                return self.orders[prefixed_id]
        
        # Try partial match (last 6 digits)
        for oid, order in self.orders.items():
            if order_id in oid or oid.endswith(order_id):
                return order
        
        return None
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status summary.
        
        Args:
            order_id: The order ID
            
        Returns:
            Status dictionary with order info
        """
        order = self.get_order(order_id)
        
        if not order:
            return {
                "found": False,
                "message": f"Order '{order_id}' not found. Please check the order number and try again.",
                "suggestion": "Order numbers start with 'TM-' followed by the year and a 6-digit number (e.g., TM-2024-001234)"
            }
        
        # Build status response
        status = {
            "found": True,
            "order_id": order['order_id'],
            "status": order['status'],
            "order_date": order['order_date'],
            "items": [
                {
                    "name": item['name'],
                    "quantity": item['quantity'],
                    "price": item['price']
                }
                for item in order['items']
            ],
            "total": order['total'],
            "shipping_method": order['shipping_method'],
            "estimated_delivery": order['estimated_delivery'],
            "actual_delivery": order.get('actual_delivery')
        }
        
        # Add tracking info if available
        if order.get('tracking_number'):
            carrier = order.get('carrier', 'Unknown')
            tracking_url = self.carriers.get(carrier, {}).get('tracking_url', '')
            
            status['tracking'] = {
                "number": order['tracking_number'],
                "carrier": carrier,
                "tracking_url": f"{tracking_url}{order['tracking_number']}" if tracking_url else None
            }
        
        # Add return info if applicable
        if order['status'] in ['Return Initiated', 'Refund Processed']:
            status['return_info'] = {
                "return_tracking": order.get('return_tracking'),
                "return_status": order.get('return_status'),
                "return_reason": order.get('return_reason')
            }
            if order.get('refund_amount'):
                status['return_info']['refund_amount'] = order['refund_amount']
                status['return_info']['refund_date'] = order.get('refund_date')
        
        # Add status history
        status['status_history'] = order.get('status_history', [])
        
        return status
    
    def get_tracking_info(self, order_id: str) -> Dict[str, Any]:
        """
        Get detailed tracking information for an order.
        
        Args:
            order_id: The order ID
            
        Returns:
            Tracking information dictionary
        """
        order = self.get_order(order_id)
        
        if not order:
            return {
                "found": False,
                "message": f"Order '{order_id}' not found."
            }
        
        if not order.get('tracking_number'):
            return {
                "found": True,
                "order_id": order['order_id'],
                "status": order['status'],
                "message": "Tracking number not yet available. Your order is being prepared for shipment.",
                "estimated_ship_date": "Within 1-2 business days"
            }
        
        carrier = order.get('carrier', 'Unknown')
        carrier_info = self.carriers.get(carrier, {})
        
        tracking = {
            "found": True,
            "order_id": order['order_id'],
            "tracking_number": order['tracking_number'],
            "carrier": carrier,
            "carrier_phone": carrier_info.get('phone'),
            "tracking_url": f"{carrier_info.get('tracking_url', '')}{order['tracking_number']}",
            "current_status": order['status'],
            "shipping_method": order['shipping_method'],
            "estimated_delivery": order['estimated_delivery'],
            "actual_delivery": order.get('actual_delivery'),
            "status_history": order.get('status_history', [])
        }
        
        # Add shipping address (partially masked for privacy)
        addr = order.get('shipping_address', {})
        if addr:
            tracking['shipping_to'] = {
                "city": addr.get('city'),
                "state": addr.get('state'),
                "zip": addr.get('zip')
            }
        
        return tracking
    
    def search_orders_by_email(self, email: str) -> List[Dict]:
        """
        Search orders by customer email.
        
        Args:
            email: Customer email address
            
        Returns:
            List of matching orders (summary)
        """
        email = email.lower().strip()
        matching_orders = []
        
        for order in self.orders.values():
            if order.get('customer_email', '').lower() == email:
                matching_orders.append({
                    "order_id": order['order_id'],
                    "order_date": order['order_date'],
                    "status": order['status'],
                    "total": order['total'],
                    "items_count": len(order['items'])
                })
        
        return matching_orders
    
    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        """Alias for get_order for API compatibility."""
        return self.get_order(order_id)
    
    def get_order_by_tracking(self, tracking_number: str) -> Optional[Dict]:
        """
        Get order by tracking number.
        
        Args:
            tracking_number: The shipment tracking number
            
        Returns:
            Order dictionary or None if not found
        """
        tracking_number = tracking_number.upper().strip()
        
        for order in self.orders.values():
            order_tracking = order.get('tracking_number', '').upper()
            if order_tracking == tracking_number:
                return order
        
        return None
    
    def get_orders_by_email(self, email: str) -> List[Dict]:
        """
        Get full order details by customer email.
        
        Args:
            email: Customer email address
            
        Returns:
            List of full order dictionaries
        """
        email = email.lower().strip()
        matching_orders = []
        
        for order in self.orders.values():
            order_email = order.get('email', order.get('customer_email', '')).lower()
            if order_email == email:
                matching_orders.append(order)
        
        return matching_orders
    
    def search_orders(self, query: str) -> List[Dict]:
        """
        Search orders by various fields (order ID, tracking number, name, email).
        
        Args:
            query: Search query string
            
        Returns:
            List of matching orders
        """
        query = query.lower().strip()
        matching_orders = []
        
        for order in self.orders.values():
            # Check various fields for match
            order_id = order.get('order_id', '').lower()
            tracking = order.get('tracking_number', '').lower()
            name = order.get('customer_name', '').lower()
            email = order.get('email', order.get('customer_email', '')).lower()
            
            if (query in order_id or 
                query in tracking or 
                query in name or 
                query in email):
                matching_orders.append(order)
        
        return matching_orders
    
    def get_order_status_summary(self, order_id: str) -> Optional[Dict]:
        """
        Get a human-readable status summary for an order.
        
        Args:
            order_id: The order ID
            
        Returns:
            Dictionary with status summary or None if not found
        """
        order = self.get_order(order_id)
        if not order:
            return None
        
        status = order.get('status', 'unknown').lower()
        
        # Create human-readable summary
        status_messages = {
            'processing': "Your order is being processed and will ship soon.",
            'shipped': f"Your order has been shipped! Tracking: {order.get('tracking_number', 'N/A')}",
            'out_for_delivery': "Your order is out for delivery today!",
            'delivered': "Your order has been delivered.",
            'cancelled': "This order has been cancelled.",
            'returned': "This order has been returned."
        }
        
        message = status_messages.get(status, f"Order status: {status}")
        
        return {
            "order_id": order['order_id'],
            "status": status,
            "status_message": message,
            "tracking_number": order.get('tracking_number'),
            "carrier": order.get('carrier'),
            "estimated_delivery": order.get('estimated_delivery'),
            "last_updated": order.get('status_updated', order.get('order_date'))
        }
    
    def get_all_order_ids(self) -> List[str]:
        """Get list of all order IDs for reference."""
        return list(self.orders.keys())
    
    def format_order_for_chat(self, order_id: str) -> str:
        """
        Format order information for chat response.
        
        Args:
            order_id: The order ID
            
        Returns:
            Formatted string for chat display
        """
        status = self.get_order_status(order_id)
        
        if not status['found']:
            return status['message']
        
        lines = [
            f"📦 **Order {status['order_id']}**",
            f"",
            f"**Status:** {status['status']}",
            f"**Order Date:** {status['order_date']}",
            f"**Total:** ${status['total']:.2f}",
            f"",
            f"**Items:**"
        ]
        
        for item in status['items']:
            lines.append(f"  • {item['name']} (x{item['quantity']}) - ${item['price']:.2f}")
        
        lines.append(f"")
        lines.append(f"**Shipping:** {status['shipping_method']}")
        lines.append(f"**Estimated Delivery:** {status['estimated_delivery']}")
        
        if status.get('actual_delivery'):
            lines.append(f"**Delivered:** {status['actual_delivery']}")
        
        if status.get('tracking'):
            lines.append(f"")
            lines.append(f"**Tracking:**")
            lines.append(f"  • Carrier: {status['tracking']['carrier']}")
            lines.append(f"  • Tracking #: {status['tracking']['number']}")
            if status['tracking'].get('tracking_url'):
                lines.append(f"  • Track at: {status['tracking']['tracking_url']}")
        
        if status.get('return_info'):
            lines.append(f"")
            lines.append(f"**Return Information:**")
            lines.append(f"  • Status: {status['return_info'].get('return_status', 'N/A')}")
            if status['return_info'].get('refund_amount'):
                lines.append(f"  • Refund: ${status['return_info']['refund_amount']:.2f}")
        
        return "\n".join(lines)


# Create singleton instance
order_manager = OrderManager()


def get_order_manager() -> OrderManager:
    """Get the order manager instance."""
    return order_manager
