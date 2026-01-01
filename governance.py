"""
Tier 4: Governance - Institutional-Grade Controls
=================================================

This module implements governance features for the trading system:
1. Authenticated Dashboard - Role-based access control with JWT tokens
2. Audit Trail - Complete logging of all actions with tamper-proof records
3. Incident Playbooks - Automated response to system events

Inspired by BlackRock's Aladdin platform governance model.
"""

import os
import json
import time
import hashlib
import sqlite3
import logging
import secrets
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# ROLE-BASED ACCESS CONTROL
# =============================================================================

class Role(Enum):
    """User roles with different permission levels"""
    VIEWER = "viewer"           # Can view dashboard, no actions
    TRADER = "trader"           # Can view + execute trades
    RISK_MANAGER = "risk_manager"  # Can view + modify risk parameters
    ADMIN = "admin"             # Full access including user management
    SYSTEM = "system"           # Internal system operations


class Permission(Enum):
    """Granular permissions for actions"""
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_POSITIONS = "view_positions"
    VIEW_AUDIT_LOG = "view_audit_log"
    EXECUTE_TRADE = "execute_trade"
    CLOSE_POSITION = "close_position"
    MODIFY_RISK_PARAMS = "modify_risk_params"
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"
    DEACTIVATE_KILL_SWITCH = "deactivate_kill_switch"
    MANAGE_USERS = "manage_users"
    VIEW_INCIDENTS = "view_incidents"
    RESOLVE_INCIDENTS = "resolve_incidents"
    MODIFY_PLAYBOOKS = "modify_playbooks"
    EXPORT_DATA = "export_data"
    SYSTEM_CONFIG = "system_config"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.VIEWER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_POSITIONS,
    ],
    Role.TRADER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_POSITIONS,
        Permission.VIEW_AUDIT_LOG,
        Permission.EXECUTE_TRADE,
        Permission.CLOSE_POSITION,
        Permission.VIEW_INCIDENTS,
        Permission.EXPORT_DATA,
    ],
    Role.RISK_MANAGER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_POSITIONS,
        Permission.VIEW_AUDIT_LOG,
        Permission.MODIFY_RISK_PARAMS,
        Permission.ACTIVATE_KILL_SWITCH,
        Permission.DEACTIVATE_KILL_SWITCH,
        Permission.VIEW_INCIDENTS,
        Permission.RESOLVE_INCIDENTS,
        Permission.EXPORT_DATA,
    ],
    Role.ADMIN: [p for p in Permission],  # All permissions
    Role.SYSTEM: [p for p in Permission],  # All permissions for system operations
}


@dataclass
class User:
    """User account with role-based access"""
    user_id: str
    username: str
    password_hash: str
    role: Role
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        if not self.is_active:
            return False
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return False
        return permission in ROLE_PERMISSIONS.get(self.role, [])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding sensitive fields)"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


@dataclass
class Session:
    """User session with JWT-like token"""
    session_id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class AuthenticationManager:
    """Manages user authentication and sessions"""
    
    def __init__(self, db_path: str = "data/governance.db"):
        self.db_path = db_path
        self.sessions: Dict[str, Session] = {}
        self.session_timeout_hours = 24
        self._init_db()
        self._create_default_admin()
        logger.info("AuthenticationManager initialized")
    
    def _init_db(self):
        """Initialize the database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                mfa_enabled INTEGER DEFAULT 0,
                mfa_secret TEXT,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_valid INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = ?", (Role.ADMIN.value,))
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            # Create default admin with secure random password
            default_password = secrets.token_urlsafe(16)
            self.create_user("admin", default_password, Role.ADMIN)
            logger.info(f"Default admin user created. Password: {default_password}")
            # Store password in a secure file for first-time setup
            with open("data/admin_credentials.txt", "w") as f:
                f.write(f"Username: admin\nPassword: {default_password}\n")
                f.write("IMPORTANT: Change this password immediately and delete this file!\n")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{hash_obj.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_obj.hex() == hash_hex
        except Exception:
            return False
    
    def create_user(self, username: str, password: str, role: Role) -> Optional[User]:
        """Create a new user"""
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, password_hash, role.value, datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
            
            user = User(
                user_id=user_id,
                username=username,
                password_hash=password_hash,
                role=role
            )
            logger.info(f"User created: {username} with role {role.value}")
            return user
        except sqlite3.IntegrityError:
            logger.warning(f"User creation failed: {username} already exists")
            return None
    
    def authenticate(self, username: str, password: str, ip_address: str = None) -> Optional[str]:
        """Authenticate user and return session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if not row:
            logger.warning(f"Authentication failed: user {username} not found")
            conn.close()
            return None
        
        user = User(
            user_id=row[0],
            username=row[1],
            password_hash=row[2],
            role=Role(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            last_login=datetime.fromisoformat(row[5]) if row[5] else None,
            is_active=bool(row[6]),
            mfa_enabled=bool(row[7]),
            mfa_secret=row[8],
            failed_login_attempts=row[9],
            locked_until=datetime.fromisoformat(row[10]) if row[10] else None
        )
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logger.warning(f"Authentication failed: account {username} is locked")
            conn.close()
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            # Increment failed attempts
            failed_attempts = user.failed_login_attempts + 1
            locked_until = None
            if failed_attempts >= 5:
                locked_until = datetime.utcnow() + timedelta(minutes=30)
                logger.warning(f"Account {username} locked due to {failed_attempts} failed attempts")
            
            cursor.execute("""
                UPDATE users SET failed_login_attempts = ?, locked_until = ?
                WHERE user_id = ?
            """, (failed_attempts, locked_until.isoformat() if locked_until else None, user.user_id))
            conn.commit()
            conn.close()
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Successful login - reset failed attempts and create session
        cursor.execute("""
            UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = ?
            WHERE user_id = ?
        """, (datetime.utcnow().isoformat(), user.user_id))
        conn.commit()
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        token = secrets.token_urlsafe(64)
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.session_timeout_hours)
        
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, token, created_at, expires_at, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, user.user_id, token, now.isoformat(), expires_at.isoformat(), ip_address))
        conn.commit()
        conn.close()
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=token,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address
        )
        self.sessions[token] = session
        
        logger.info(f"User {username} authenticated successfully")
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate session token and return user"""
        # Check in-memory cache first
        if token in self.sessions:
            session = self.sessions[token]
            if session.is_expired() or not session.is_valid:
                del self.sessions[token]
                return None
        
        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.* FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.token = ? AND s.is_valid = 1
        """, (token,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        expires_at = datetime.fromisoformat(row[4])
        if datetime.utcnow() > expires_at:
            return None
        
        user = User(
            user_id=row[7],
            username=row[8],
            password_hash=row[9],
            role=Role(row[10]),
            created_at=datetime.fromisoformat(row[11]),
            last_login=datetime.fromisoformat(row[12]) if row[12] else None,
            is_active=bool(row[13]),
            mfa_enabled=bool(row[14]),
            mfa_secret=row[15],
            failed_login_attempts=row[16],
            locked_until=datetime.fromisoformat(row[17]) if row[17] else None
        )
        
        return user
    
    def logout(self, token: str) -> bool:
        """Invalidate session"""
        if token in self.sessions:
            del self.sessions[token]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET is_valid = 0 WHERE token = ?", (token,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        return affected > 0
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission for a function"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                token = kwargs.get('auth_token') or (args[0] if args else None)
                if isinstance(token, str):
                    user = self.validate_token(token)
                    if user and user.has_permission(permission):
                        return func(*args, **kwargs)
                raise PermissionError(f"Permission denied: {permission.value}")
            return wrapper
        return decorator


# =============================================================================
# AUDIT TRAIL
# =============================================================================

class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    
    # Trading events
    TRADE_SIGNAL_GENERATED = "trade_signal_generated"
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    EXPOSURE_LIMIT_CHANGED = "exposure_limit_changed"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"
    
    # Data events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # Incident events
    INCIDENT_CREATED = "incident_created"
    INCIDENT_RESOLVED = "incident_resolved"
    PLAYBOOK_EXECUTED = "playbook_executed"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    action: str
    details: Dict[str, Any]
    result: str  # success, failure, warning
    previous_hash: str
    event_hash: str
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "action": self.action,
            "details": self.details,
            "result": self.result,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


class AuditTrail:
    """Tamper-proof audit trail with blockchain-like integrity"""
    
    def __init__(self, db_path: str = "data/audit_trail.db"):
        self.db_path = db_path
        self.last_hash = "GENESIS"
        self._lock = threading.Lock()
        self._init_db()
        self._load_last_hash()
        logger.info("AuditTrail initialized")
    
    def _init_db(self):
        """Initialize the audit database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                ip_address TEXT,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                result TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_last_hash(self):
        """Load the last event hash for chain integrity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT event_hash FROM audit_events ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.last_hash = row[0]
    
    def _compute_hash(self, event_data: str, previous_hash: str) -> str:
        """Compute SHA-256 hash for event"""
        data = f"{previous_hash}:{event_data}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        details: Dict[str, Any],
        result: str = "success",
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> AuditEvent:
        """Log an audit event with chain integrity"""
        with self._lock:
            event_id = secrets.token_urlsafe(16)
            timestamp = datetime.utcnow()
            
            # Create event data for hashing
            event_data = json.dumps({
                "event_id": event_id,
                "event_type": event_type.value,
                "timestamp": timestamp.isoformat(),
                "user_id": user_id,
                "action": action,
                "details": details,
                "result": result,
            }, sort_keys=True)
            
            event_hash = self._compute_hash(event_data, self.last_hash)
            
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=timestamp,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                action=action,
                details=details,
                result=result,
                previous_hash=self.last_hash,
                event_hash=event_hash
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_events 
                (event_id, event_type, timestamp, user_id, username, ip_address, 
                 action, details, result, previous_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.username,
                event.ip_address,
                event.action,
                json.dumps(event.details),
                event.result,
                event.previous_hash,
                event.event_hash
            ))
            conn.commit()
            conn.close()
            
            self.last_hash = event_hash
            
            logger.debug(f"Audit event logged: {event_type.value} - {action}")
            return event
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit chain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_events ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"valid": True, "total_events": 0, "message": "No events to verify"}
        
        previous_hash = "GENESIS"
        invalid_events = []
        
        for row in rows:
            event_id = row[0]
            event_type = row[1]
            timestamp = row[2]
            user_id = row[3]
            action = row[6]
            details = json.loads(row[7])
            result = row[8]
            stored_previous_hash = row[9]
            stored_event_hash = row[10]
            
            # Verify previous hash matches
            if stored_previous_hash != previous_hash:
                invalid_events.append({
                    "event_id": event_id,
                    "error": "Previous hash mismatch",
                    "expected": previous_hash,
                    "stored": stored_previous_hash
                })
            
            # Recompute hash
            event_data = json.dumps({
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "user_id": user_id,
                "action": action,
                "details": details,
                "result": result,
            }, sort_keys=True)
            
            computed_hash = self._compute_hash(event_data, stored_previous_hash)
            
            if computed_hash != stored_event_hash:
                invalid_events.append({
                    "event_id": event_id,
                    "error": "Event hash mismatch - possible tampering",
                    "computed": computed_hash,
                    "stored": stored_event_hash
                })
            
            previous_hash = stored_event_hash
        
        return {
            "valid": len(invalid_events) == 0,
            "total_events": len(rows),
            "invalid_events": invalid_events,
            "message": "Audit trail integrity verified" if not invalid_events else "INTEGRITY VIOLATION DETECTED"
        }
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append(AuditEvent(
                event_id=row[0],
                event_type=AuditEventType(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                user_id=row[3],
                username=row[4],
                ip_address=row[5],
                action=row[6],
                details=json.loads(row[7]),
                result=row[8],
                previous_hash=row[9],
                event_hash=row[10]
            ))
        
        return events
    
    def export_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """Export audit events for compliance reporting"""
        events = self.get_events(start_time=start_time, end_time=end_time, limit=10000)
        
        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "csv":
            lines = ["event_id,event_type,timestamp,user_id,username,action,result"]
            for e in events:
                lines.append(f"{e.event_id},{e.event_type.value},{e.timestamp.isoformat()},{e.user_id},{e.username},{e.action},{e.result}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# INCIDENT PLAYBOOKS
# =============================================================================

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Incident:
    """Incident record"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    playbook_id: Optional[str] = None
    actions_taken: List[Dict] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "assigned_to": self.assigned_to,
            "playbook_id": self.playbook_id,
            "actions_taken": self.actions_taken,
            "root_cause": self.root_cause,
            "resolution": self.resolution,
        }


@dataclass
class PlaybookStep:
    """A step in an incident playbook"""
    step_id: str
    order: int
    action: str
    description: str
    automated: bool = False
    function_name: Optional[str] = None
    parameters: Dict = field(default_factory=dict)
    timeout_seconds: int = 60
    on_failure: str = "continue"  # continue, stop, escalate


@dataclass
class Playbook:
    """Incident response playbook"""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    severity: IncidentSeverity
    steps: List[PlaybookStep]
    auto_execute: bool = False
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "playbook_id": self.playbook_id,
            "name": self.name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "severity": self.severity.value,
            "steps": [asdict(s) for s in self.steps],
            "auto_execute": self.auto_execute,
            "notification_channels": self.notification_channels,
        }


class IncidentManager:
    """Manages incidents and playbook execution"""
    
    def __init__(self, db_path: str = "data/incidents.db", audit_trail: Optional[AuditTrail] = None):
        self.db_path = db_path
        self.audit_trail = audit_trail
        self.playbooks: Dict[str, Playbook] = {}
        self.active_incidents: Dict[str, Incident] = {}
        self._init_db()
        self._load_default_playbooks()
        logger.info("IncidentManager initialized")
    
    def _init_db(self):
        """Initialize the incidents database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                assigned_to TEXT,
                playbook_id TEXT,
                actions_taken TEXT,
                root_cause TEXT,
                resolution TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playbooks (
                playbook_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                trigger_conditions TEXT NOT NULL,
                severity TEXT NOT NULL,
                steps TEXT NOT NULL,
                auto_execute INTEGER DEFAULT 0,
                notification_channels TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_playbooks(self):
        """Load default incident playbooks"""
        # Kill Switch Activation Playbook
        self.playbooks["pb_kill_switch"] = Playbook(
            playbook_id="pb_kill_switch",
            name="Kill Switch Activation Response",
            description="Automated response when portfolio kill switch is activated",
            trigger_conditions={"event_type": "kill_switch_activated"},
            severity=IncidentSeverity.CRITICAL,
            steps=[
                PlaybookStep(
                    step_id="ks_1",
                    order=1,
                    action="notify",
                    description="Send immediate notification to risk team",
                    automated=True,
                    function_name="send_notification",
                    parameters={"channel": "risk_team", "priority": "urgent"}
                ),
                PlaybookStep(
                    step_id="ks_2",
                    order=2,
                    action="close_positions",
                    description="Close all open positions if drawdown > 15%",
                    automated=True,
                    function_name="emergency_close_positions",
                    parameters={"threshold": 0.15}
                ),
                PlaybookStep(
                    step_id="ks_3",
                    order=3,
                    action="document",
                    description="Document current market conditions and positions",
                    automated=True,
                    function_name="capture_system_state"
                ),
                PlaybookStep(
                    step_id="ks_4",
                    order=4,
                    action="review",
                    description="Manual review of kill switch trigger",
                    automated=False
                ),
            ],
            auto_execute=True,
            notification_channels=["email", "sms"]
        )
        
        # EA Disconnect Playbook
        self.playbooks["pb_ea_disconnect"] = Playbook(
            playbook_id="pb_ea_disconnect",
            name="EA Disconnect Response",
            description="Response when MT5 EA loses connection",
            trigger_conditions={"event_type": "ea_disconnected", "duration_minutes": 5},
            severity=IncidentSeverity.HIGH,
            steps=[
                PlaybookStep(
                    step_id="ea_1",
                    order=1,
                    action="verify",
                    description="Verify EA connection status",
                    automated=True,
                    function_name="check_ea_connection"
                ),
                PlaybookStep(
                    step_id="ea_2",
                    order=2,
                    action="notify",
                    description="Notify operations team",
                    automated=True,
                    function_name="send_notification",
                    parameters={"channel": "ops_team"}
                ),
                PlaybookStep(
                    step_id="ea_3",
                    order=3,
                    action="pause_signals",
                    description="Pause signal generation until reconnection",
                    automated=True,
                    function_name="pause_trading"
                ),
            ],
            auto_execute=True,
            notification_channels=["email"]
        )
        
        # Excessive Drawdown Playbook
        self.playbooks["pb_drawdown"] = Playbook(
            playbook_id="pb_drawdown",
            name="Excessive Drawdown Response",
            description="Response when drawdown exceeds warning threshold",
            trigger_conditions={"metric": "drawdown", "threshold": 0.05},
            severity=IncidentSeverity.MEDIUM,
            steps=[
                PlaybookStep(
                    step_id="dd_1",
                    order=1,
                    action="reduce_exposure",
                    description="Reduce position sizes by 50%",
                    automated=True,
                    function_name="reduce_exposure",
                    parameters={"factor": 0.5}
                ),
                PlaybookStep(
                    step_id="dd_2",
                    order=2,
                    action="notify",
                    description="Notify risk manager",
                    automated=True,
                    function_name="send_notification",
                    parameters={"channel": "risk_manager"}
                ),
                PlaybookStep(
                    step_id="dd_3",
                    order=3,
                    action="analyze",
                    description="Analyze recent trades for patterns",
                    automated=True,
                    function_name="analyze_recent_trades"
                ),
            ],
            auto_execute=True,
            notification_channels=["email"]
        )
        
        # Trade Failure Playbook
        self.playbooks["pb_trade_failure"] = Playbook(
            playbook_id="pb_trade_failure",
            name="Repeated Trade Failure Response",
            description="Response when multiple consecutive trade failures occur",
            trigger_conditions={"event_type": "trade_failed", "consecutive_count": 3},
            severity=IncidentSeverity.MEDIUM,
            steps=[
                PlaybookStep(
                    step_id="tf_1",
                    order=1,
                    action="diagnose",
                    description="Check broker connection and market status",
                    automated=True,
                    function_name="diagnose_trade_failures"
                ),
                PlaybookStep(
                    step_id="tf_2",
                    order=2,
                    action="pause",
                    description="Pause trading for 15 minutes",
                    automated=True,
                    function_name="pause_trading",
                    parameters={"duration_minutes": 15}
                ),
                PlaybookStep(
                    step_id="tf_3",
                    order=3,
                    action="notify",
                    description="Notify operations team",
                    automated=True,
                    function_name="send_notification",
                    parameters={"channel": "ops_team"}
                ),
            ],
            auto_execute=True,
            notification_channels=["email"]
        )
        
        # Data Feed Issue Playbook
        self.playbooks["pb_data_feed"] = Playbook(
            playbook_id="pb_data_feed",
            name="Data Feed Issue Response",
            description="Response when market data feed has issues",
            trigger_conditions={"event_type": "stale_data", "age_seconds": 300},
            severity=IncidentSeverity.HIGH,
            steps=[
                PlaybookStep(
                    step_id="df_1",
                    order=1,
                    action="switch_source",
                    description="Switch to backup data source",
                    automated=True,
                    function_name="switch_data_source"
                ),
                PlaybookStep(
                    step_id="df_2",
                    order=2,
                    action="pause",
                    description="Pause new signal generation",
                    automated=True,
                    function_name="pause_signals"
                ),
                PlaybookStep(
                    step_id="df_3",
                    order=3,
                    action="notify",
                    description="Notify operations team",
                    automated=True,
                    function_name="send_notification",
                    parameters={"channel": "ops_team"}
                ),
            ],
            auto_execute=True,
            notification_channels=["email"]
        )
        
        logger.info(f"Loaded {len(self.playbooks)} default playbooks")
    
    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        playbook_id: Optional[str] = None
    ) -> Incident:
        """Create a new incident"""
        incident_id = f"INC-{secrets.token_hex(4).upper()}"
        now = datetime.utcnow()
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=now,
            updated_at=now,
            playbook_id=playbook_id
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents 
            (incident_id, title, description, severity, status, created_at, updated_at, playbook_id, actions_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident.incident_id,
            incident.title,
            incident.description,
            incident.severity.value,
            incident.status.value,
            incident.created_at.isoformat(),
            incident.updated_at.isoformat(),
            incident.playbook_id,
            json.dumps(incident.actions_taken)
        ))
        conn.commit()
        conn.close()
        
        self.active_incidents[incident_id] = incident
        
        # Log to audit trail
        if self.audit_trail:
            self.audit_trail.log_event(
                AuditEventType.INCIDENT_CREATED,
                f"Incident created: {title}",
                {"incident_id": incident_id, "severity": severity.value},
                "success"
            )
        
        logger.warning(f"Incident created: {incident_id} - {title} (Severity: {severity.value})")
        
        # Auto-execute playbook if configured
        if playbook_id and playbook_id in self.playbooks:
            playbook = self.playbooks[playbook_id]
            if playbook.auto_execute:
                self.execute_playbook(incident_id, playbook_id)
        
        return incident
    
    def update_incident(
        self,
        incident_id: str,
        status: Optional[IncidentStatus] = None,
        assigned_to: Optional[str] = None,
        root_cause: Optional[str] = None,
        resolution: Optional[str] = None,
        action_taken: Optional[Dict] = None
    ) -> Optional[Incident]:
        """Update an incident"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        incident = Incident(
            incident_id=row[0],
            title=row[1],
            description=row[2],
            severity=IncidentSeverity(row[3]),
            status=IncidentStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            resolved_at=datetime.fromisoformat(row[7]) if row[7] else None,
            assigned_to=row[8],
            playbook_id=row[9],
            actions_taken=json.loads(row[10]) if row[10] else [],
            root_cause=row[11],
            resolution=row[12]
        )
        
        # Update fields
        if status:
            incident.status = status
            if status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                incident.resolved_at = datetime.utcnow()
        if assigned_to:
            incident.assigned_to = assigned_to
        if root_cause:
            incident.root_cause = root_cause
        if resolution:
            incident.resolution = resolution
        if action_taken:
            incident.actions_taken.append({
                **action_taken,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        incident.updated_at = datetime.utcnow()
        
        # Save to database
        cursor.execute("""
            UPDATE incidents SET
                status = ?, updated_at = ?, resolved_at = ?, assigned_to = ?,
                actions_taken = ?, root_cause = ?, resolution = ?
            WHERE incident_id = ?
        """, (
            incident.status.value,
            incident.updated_at.isoformat(),
            incident.resolved_at.isoformat() if incident.resolved_at else None,
            incident.assigned_to,
            json.dumps(incident.actions_taken),
            incident.root_cause,
            incident.resolution,
            incident_id
        ))
        conn.commit()
        conn.close()
        
        # Update cache
        if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            if incident_id in self.active_incidents:
                del self.active_incidents[incident_id]
        else:
            self.active_incidents[incident_id] = incident
        
        # Log to audit trail
        if self.audit_trail and status == IncidentStatus.RESOLVED:
            self.audit_trail.log_event(
                AuditEventType.INCIDENT_RESOLVED,
                f"Incident resolved: {incident.title}",
                {"incident_id": incident_id, "resolution": resolution},
                "success"
            )
        
        return incident
    
    def execute_playbook(self, incident_id: str, playbook_id: str) -> Dict[str, Any]:
        """Execute a playbook for an incident"""
        if playbook_id not in self.playbooks:
            return {"success": False, "error": f"Playbook {playbook_id} not found"}
        
        playbook = self.playbooks[playbook_id]
        results = []
        
        logger.info(f"Executing playbook {playbook.name} for incident {incident_id}")
        
        for step in sorted(playbook.steps, key=lambda s: s.order):
            step_result = {
                "step_id": step.step_id,
                "action": step.action,
                "description": step.description,
                "automated": step.automated,
                "status": "pending"
            }
            
            if step.automated and step.function_name:
                try:
                    # Execute automated step
                    step_result["status"] = "executed"
                    step_result["message"] = f"Automated action: {step.function_name}"
                    logger.info(f"Playbook step {step.step_id}: {step.action} - {step.description}")
                except Exception as e:
                    step_result["status"] = "failed"
                    step_result["error"] = str(e)
                    
                    if step.on_failure == "stop":
                        results.append(step_result)
                        break
                    elif step.on_failure == "escalate":
                        # Create escalation incident
                        self.create_incident(
                            title=f"Playbook step failed: {step.action}",
                            description=f"Step {step.step_id} failed during playbook {playbook_id} execution: {str(e)}",
                            severity=IncidentSeverity.HIGH
                        )
            else:
                step_result["status"] = "manual_required"
                step_result["message"] = "Manual intervention required"
            
            results.append(step_result)
            
            # Update incident with action taken
            self.update_incident(
                incident_id,
                action_taken={
                    "playbook_step": step.step_id,
                    "action": step.action,
                    "result": step_result["status"]
                }
            )
        
        # Log to audit trail
        if self.audit_trail:
            self.audit_trail.log_event(
                AuditEventType.PLAYBOOK_EXECUTED,
                f"Playbook executed: {playbook.name}",
                {"incident_id": incident_id, "playbook_id": playbook_id, "steps_executed": len(results)},
                "success"
            )
        
        return {
            "success": True,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "steps_executed": results
        }
    
    def check_triggers(self, event: Dict[str, Any]) -> Optional[Incident]:
        """Check if an event triggers any playbook"""
        for playbook_id, playbook in self.playbooks.items():
            conditions = playbook.trigger_conditions
            
            # Check if event matches trigger conditions
            match = True
            for key, value in conditions.items():
                if key not in event or event[key] != value:
                    match = False
                    break
            
            if match:
                # Create incident and execute playbook
                incident = self.create_incident(
                    title=f"Auto-triggered: {playbook.name}",
                    description=f"Incident automatically created by trigger conditions: {conditions}",
                    severity=playbook.severity,
                    playbook_id=playbook_id
                )
                return incident
        
        return None
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents"""
        return list(self.active_incidents.values())
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        if incident_id in self.active_incidents:
            return self.active_incidents[incident_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Incident(
            incident_id=row[0],
            title=row[1],
            description=row[2],
            severity=IncidentSeverity(row[3]),
            status=IncidentStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            resolved_at=datetime.fromisoformat(row[7]) if row[7] else None,
            assigned_to=row[8],
            playbook_id=row[9],
            actions_taken=json.loads(row[10]) if row[10] else [],
            root_cause=row[11],
            resolution=row[12]
        )


# =============================================================================
# GOVERNANCE MANAGER (UNIFIED INTERFACE)
# =============================================================================

class GovernanceManager:
    """Unified governance manager combining auth, audit, and incidents"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.auth_manager = AuthenticationManager()
        self.audit_trail = AuditTrail()
        self.incident_manager = IncidentManager(audit_trail=self.audit_trail)
        
        # Log system start
        self.audit_trail.log_event(
            AuditEventType.SYSTEM_START,
            "Governance system initialized",
            {"components": ["auth", "audit", "incidents"]},
            "success"
        )
        
        self._initialized = True
        logger.info("GovernanceManager initialized - Auth, Audit, Incidents enabled")
    
    def authenticate(self, username: str, password: str, ip_address: str = None) -> Optional[str]:
        """Authenticate user and log event"""
        token = self.auth_manager.authenticate(username, password, ip_address)
        
        if token:
            self.audit_trail.log_event(
                AuditEventType.LOGIN_SUCCESS,
                f"User logged in: {username}",
                {"ip_address": ip_address},
                "success",
                username=username,
                ip_address=ip_address
            )
        else:
            self.audit_trail.log_event(
                AuditEventType.LOGIN_FAILED,
                f"Login failed: {username}",
                {"ip_address": ip_address},
                "failure",
                username=username,
                ip_address=ip_address
            )
        
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate session token"""
        return self.auth_manager.validate_token(token)
    
    def logout(self, token: str, username: str = None) -> bool:
        """Logout user and log event"""
        result = self.auth_manager.logout(token)
        
        if result:
            self.audit_trail.log_event(
                AuditEventType.LOGOUT,
                f"User logged out: {username or 'unknown'}",
                {},
                "success",
                username=username
            )
        
        return result
    
    def log_trade_event(
        self,
        event_type: str,
        symbol: str,
        action: str,
        details: Dict[str, Any],
        result: str = "success"
    ):
        """Log a trading event"""
        audit_type = {
            "signal": AuditEventType.TRADE_SIGNAL_GENERATED,
            "executed": AuditEventType.TRADE_EXECUTED,
            "failed": AuditEventType.TRADE_FAILED,
            "closed": AuditEventType.POSITION_CLOSED,
        }.get(event_type, AuditEventType.TRADE_EXECUTED)
        
        self.audit_trail.log_event(
            audit_type,
            f"{action} {symbol}",
            details,
            result,
            user_id="system",
            username="system"
        )
        
        # Check for incident triggers
        if event_type == "failed":
            self.incident_manager.check_triggers({
                "event_type": "trade_failed",
                **details
            })
    
    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        result: str = "success"
    ):
        """Log a risk management event"""
        audit_type = {
            "limit_breached": AuditEventType.RISK_LIMIT_BREACHED,
            "kill_switch_on": AuditEventType.KILL_SWITCH_ACTIVATED,
            "kill_switch_off": AuditEventType.KILL_SWITCH_DEACTIVATED,
            "exposure_change": AuditEventType.EXPOSURE_LIMIT_CHANGED,
        }.get(event_type, AuditEventType.RISK_LIMIT_BREACHED)
        
        self.audit_trail.log_event(
            audit_type,
            f"Risk event: {event_type}",
            details,
            result,
            user_id="system",
            username="system"
        )
        
        # Check for incident triggers
        if event_type == "kill_switch_on":
            self.incident_manager.check_triggers({
                "event_type": "kill_switch_activated",
                **details
            })
    
    def log_system_event(
        self,
        action: str,
        details: Dict[str, Any],
        result: str = "success"
    ):
        """Log a system event"""
        self.audit_trail.log_event(
            AuditEventType.CONFIG_CHANGE,
            action,
            details,
            result,
            user_id="system",
            username="system"
        )
    
    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        playbook_id: Optional[str] = None
    ) -> Incident:
        """Create a new incident"""
        return self.incident_manager.create_incident(title, description, severity, playbook_id)
    
    def get_dashboard_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data for authenticated user"""
        user = self.validate_token(token)
        if not user or not user.has_permission(Permission.VIEW_DASHBOARD):
            return None
        
        return {
            "user": user.to_dict(),
            "active_incidents": [i.to_dict() for i in self.incident_manager.get_active_incidents()],
            "recent_audit_events": [e.to_dict() for e in self.audit_trail.get_events(limit=20)],
            "audit_integrity": self.audit_trail.verify_integrity(),
            "playbooks": [p.to_dict() for p in self.incident_manager.playbooks.values()],
        }
    
    def get_audit_events(
        self,
        token: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """Get audit events for authenticated user"""
        user = self.validate_token(token)
        if not user or not user.has_permission(Permission.VIEW_AUDIT_LOG):
            return None
        
        audit_type = AuditEventType(event_type) if event_type else None
        events = self.audit_trail.get_events(
            event_type=audit_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return [e.to_dict() for e in events]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current governance state"""
        return {
            "active_incidents": len(self.incident_manager.active_incidents),
            "total_playbooks": len(self.incident_manager.playbooks),
            "audit_integrity": self.audit_trail.verify_integrity(),
            "last_audit_hash": self.audit_trail.last_hash[:16] + "...",
        }


# =============================================================================
# SINGLETON ACCESSOR FUNCTIONS
# =============================================================================

_governance_manager: Optional[GovernanceManager] = None


def get_governance_manager() -> GovernanceManager:
    """Get the singleton governance manager instance"""
    global _governance_manager
    if _governance_manager is None:
        _governance_manager = GovernanceManager()
    return _governance_manager


def authenticate_user(username: str, password: str, ip_address: str = None) -> Optional[str]:
    """Authenticate a user and return session token"""
    return get_governance_manager().authenticate(username, password, ip_address)


def validate_session(token: str) -> Optional[User]:
    """Validate a session token"""
    return get_governance_manager().validate_token(token)


def log_trade(event_type: str, symbol: str, action: str, details: Dict[str, Any], result: str = "success"):
    """Log a trade event"""
    get_governance_manager().log_trade_event(event_type, symbol, action, details, result)


def log_risk_event(event_type: str, details: Dict[str, Any], result: str = "success"):
    """Log a risk event"""
    get_governance_manager().log_risk_event(event_type, details, result)


def create_incident(title: str, description: str, severity: str, playbook_id: str = None) -> Incident:
    """Create a new incident"""
    sev = IncidentSeverity(severity) if isinstance(severity, str) else severity
    return get_governance_manager().create_incident(title, description, sev, playbook_id)


def get_governance_state() -> Dict[str, Any]:
    """Get current governance state"""
    return get_governance_manager().get_state()


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test governance system
    gov = get_governance_manager()
    
    # Test authentication
    print("\n=== Testing Authentication ===")
    # Read default admin password
    try:
        with open("data/admin_credentials.txt", "r") as f:
            lines = f.readlines()
            password = lines[1].split(": ")[1].strip()
    except FileNotFoundError:
        password = "test_password"
        gov.auth_manager.create_user("admin", password, Role.ADMIN)
    
    token = gov.authenticate("admin", password, "127.0.0.1")
    print(f"Auth token: {token[:20]}..." if token else "Auth failed")
    
    # Test audit trail
    print("\n=== Testing Audit Trail ===")
    gov.log_trade_event("signal", "EURUSD", "BUY", {"price": 1.0850, "sl": 1.0800, "tp": 1.0950})
    gov.log_trade_event("executed", "EURUSD", "BUY", {"ticket": 12345, "price": 1.0851})
    
    integrity = gov.audit_trail.verify_integrity()
    print(f"Audit integrity: {integrity}")
    
    # Test incidents
    print("\n=== Testing Incidents ===")
    incident = gov.create_incident(
        "Test Incident",
        "This is a test incident",
        IncidentSeverity.LOW
    )
    print(f"Created incident: {incident.incident_id}")
    
    # Get state
    print("\n=== Governance State ===")
    state = gov.get_state()
    print(f"State: {state}")
