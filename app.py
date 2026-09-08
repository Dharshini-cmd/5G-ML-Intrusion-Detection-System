# app.py - Flask backend for 5G IDS
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import tensorflow as tf
import os
import requests
from datetime import datetime
import ipaddress
import logging
import nmap
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
from collections import deque

load_dotenv()

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to saved_model (inside 5G_IDS_Results folder)
MODEL_PATH = os.path.join(BASE_DIR, '5G_IDS_Results', 'saved_model')

# Path to images (inside 5G_IDS_Results folder)
IMAGES_PATH = os.path.join(BASE_DIR, '5G_IDS_Results')

# API Keys
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ENGINE = "gemini-2.0-flash-exp"
MAX_ALERT_HISTORY = 100
ALERT_HISTORY = deque(maxlen=MAX_ALERT_HISTORY)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

print("="*50)
print("Loading 5G IDS Model...")
print("="*50)
print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Images path: {IMAGES_PATH}")

# Check if model path exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model path not found: {MODEL_PATH}")
    print("Please check your folder structure")
    exit(1)

# Load model and files
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'best_5g_ids_model.h5'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))

print(f"✅ Model loaded successfully!")
print(f"✅ Features: {len(feature_names)}")
print("="*50)


def check_ip_reputation(ip_address):
    """
    Enhanced IP reputation check using multiple sources
    """
    
    # Validate IP
    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        return {
            'error': 'Invalid IP address format',
            'risk_score': None
        }
    
    results = {
        'ip': ip_address,
        'risk_score': 0,
        'risk_factors': [],
        'threat_intel': {},
        'port_scan': {},
        'gemini_analysis': '',
        'reasons': [],
        'recommendations': []
    }
    
    # ============================================
    # 1. AbuseIPDB Check
    # ============================================
    if ABUSEIPDB_API_KEY:
        try:
            url = 'https://api.abuseipdb.com/api/v2/check'
            headers = {'Key': ABUSEIPDB_API_KEY, 'Accept': 'application/json'}
            params = {
                'ipAddress': ip_address,
                'maxAgeInDays': '90',
                'verbose': True
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                abuse_data = data.get('data', {})
                
                confidence = abuse_data.get('abuseConfidenceScore', 0)
                total_reports = abuse_data.get('totalReports', 0)
                
                results['threat_intel']['abuseipdb'] = {
                    'confidence': confidence,
                    'total_reports': total_reports,
                    'last_reported': abuse_data.get('lastReportedAt', 'Never'),
                    'country': abuse_data.get('countryCode', 'Unknown'),
                    'isp': abuse_data.get('isp', 'Unknown')
                }
                
                # Enhanced scoring based on multiple factors
                if confidence >= 80:
                    results['risk_factors'].append(f"CRITICAL: {confidence}% abuse confidence score")
                    results['risk_score'] += 45
                elif confidence >= 50:
                    results['risk_factors'].append(f"HIGH: {confidence}% abuse confidence score")
                    results['risk_score'] += 35
                elif confidence >= 25:
                    results['risk_factors'].append(f"MEDIUM: {confidence}% abuse confidence score")
                    results['risk_score'] += 20
                elif confidence >= 10:
                    results['risk_factors'].append(f"LOW: {confidence}% abuse confidence score")
                    results['risk_score'] += 10
                
                # Add points for multiple reports
                if total_reports > 50:
                    results['risk_score'] += 20
                    results['risk_factors'].append(f"High volume of reports ({total_reports} total)")
                elif total_reports > 20:
                    results['risk_score'] += 10
                    results['risk_factors'].append(f"Multiple reports ({total_reports} total)")
                elif total_reports > 5:
                    results['risk_score'] += 5
                    
        except Exception as e:
            logging.error(f"AbuseIPDB API error: {e}")
    
    # ============================================
    # 2. IPQuery.io (Free, no API key needed)
    # ============================================
    try:
        ipquery_response = requests.get(f'https://ipquery.io/api/{ip_address}', timeout=8)
        if ipquery_response.status_code == 200:
            ipquery_data = ipquery_response.json()
            
            results['threat_intel']['ipquery'] = ipquery_data
            
            # Check for malicious indicators
            security = ipquery_data.get('security', {})
            
            if security.get('is_vpn'):
                results['risk_factors'].append("IP is using VPN - often used to hide malicious activity")
                results['risk_score'] += 15
                
            if security.get('is_proxy'):
                results['risk_factors'].append("IP is a proxy server - may hide origin of attacks")
                results['risk_score'] += 15
                
            if security.get('is_tor'):
                results['risk_factors'].append("IP is a Tor exit node - commonly used for malicious activities")
                results['risk_score'] += 25
                
            # Check risk score from ipquery
            risk = security.get('risk_score', 0)
            if risk > 50:
                results['risk_factors'].append(f"High risk score ({risk}) from IPQuery")
                results['risk_score'] += min(risk, 30)
                
    except Exception as e:
        logging.error(f"IPQuery error: {e}")
    
    # ============================================
    # 3. Quick Port Scan (Top 20 suspicious ports)
    # ============================================
    suspicious_ports = {
        20: 'FTP-Data', 21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP',
        53: 'DNS', 80: 'HTTP', 110: 'POP3', 111: 'RPC', 135: 'RPC',
        139: 'NetBIOS', 143: 'IMAP', 443: 'HTTPS', 445: 'SMB', 993: 'IMAPS',
        995: 'POP3S', 1433: 'MSSQL', 3306: 'MySQL', 3389: 'RDP', 5432: 'PostgreSQL',
        5900: 'VNC', 6379: 'Redis', 8080: 'HTTP-Alt', 8443: 'HTTPS-Alt', 27017: 'MongoDB'
    }
    
    high_risk_ports = {445: 'SMB', 3389: 'RDP', 23: 'Telnet', 1433: 'MSSQL', 3306: 'MySQL'}
    
    try:
        nm = nmap.PortScanner()
        ports_str = ','.join(map(str, suspicious_ports.keys()))
        nm.scan(hosts=ip_address, ports=ports_str, arguments='-Pn -T4 --open', timeout=60)
        
        open_ports_found = []
        if ip_address in nm.all_hosts():
            for proto in nm[ip_address].all_protocols():
                for port in nm[ip_address][proto].keys():
                    if nm[ip_address][proto][port]['state'] == 'open':
                        service = nm[ip_address][proto][port].get('name', 'unknown')
                        version = nm[ip_address][proto][port].get('version', '')
                        open_ports_found.append({
                            'port': port,
                            'protocol': proto,
                            'service': service,
                            'version': version
                        })
                        
                        # High risk ports
                        if port in high_risk_ports:
                            results['risk_factors'].append(f"High-risk port {port}/{proto} ({high_risk_ports[port]}) is open - common target for attacks")
                            results['risk_score'] += 20
                        elif port in suspicious_ports:
                            results['risk_factors'].append(f"Port {port}/{proto} ({suspicious_ports[port]}) is exposed")
                            results['risk_score'] += 5
        
        results['port_scan']['open_ports'] = open_ports_found
        results['port_scan']['total_open'] = len(open_ports_found)
        
    except Exception as e:
        logging.error(f"Port scan error: {e}")
        results['port_scan']['error'] = str(e)
    
    # ============================================
    # 4. Check against known malicious IP lists
    # ============================================
    known_malicious_ips = {
        '45.155.205.233': 'Known C2 server',
        '185.130.5.253': 'Known malware distribution',
        '94.102.61.78': 'Known Cobalt Strike C2',
        '185.220.101.0': 'Tor exit node',
        '5.188.86.0': 'Known scanning activity',
        '45.142.120.5': 'Known DDoS botnet C2',
        '103.136.10.0': 'Known phishing host'
    }
    
    if ip_address in known_malicious_ips:
        results['risk_factors'].append(f"IP matches known threat feed: {known_malicious_ips[ip_address]}")
        results['risk_score'] += 50
    
    # ============================================
    # 5. Country-based risk scoring
    # ============================================
    high_risk_countries = ['RU', 'CN', 'KP', 'IR', 'SY', 'PK', 'NG', 'VN']
    medium_risk_countries = ['UA', 'IN', 'BR', 'MX', 'ID', 'TH', 'MY']
    
    country = results['threat_intel'].get('abuseipdb', {}).get('country', '')
    if country in high_risk_countries:
        results['risk_factors'].append(f"IP located in high-risk country ({country})")
        results['risk_score'] += 15
    elif country in medium_risk_countries:
        results['risk_factors'].append(f"IP located in medium-risk country ({country})")
        results['risk_score'] += 5
    
    # ============================================
    # 6. Gemini AI Analysis (for final verdict)
    # ============================================
    if GEMINI_API_KEY:
        try:
            prompt = f"""
You are a cybersecurity expert analyzing IP: {ip_address}

Findings:
{chr(10).join(results['risk_factors']) if results['risk_factors'] else 'No specific findings'}

Open Ports: {len(results['port_scan'].get('open_ports', []))} ports open

Based on this, provide:
1. is_malicious: "Yes" or "No" or "Suspicious"
2. risk_score: 0-100 (adjust based on findings above)
3. reasons: List specific reasons
4. recommendations: What action to take
5. summary: One-line conclusion

Return ONLY JSON:
{{"is_malicious": "", "risk_score": 0, "reasons": [], "recommendations": [], "summary": ""}}
"""
            model_gemini = genai.GenerativeModel(MODEL_ENGINE)
            response = model_gemini.generate_content(prompt)
            
            import json as json_lib
            response_text = response.text.strip()
            
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            gemini_result = json_lib.loads(response_text)
            results['gemini_analysis'] = gemini_result
            
            # Weight Gemini's score (40% of total)
            gemini_score = gemini_result.get('risk_score', 0)
            results['risk_score'] = int((results['risk_score'] * 0.6) + (gemini_score * 0.4))
            
            # Add Gemini's reasons
            for reason in gemini_result.get('reasons', []):
                if reason not in results['risk_factors']:
                    results['reasons'].append(reason)
            
            results['recommendations'] = gemini_result.get('recommendations', [])
            
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            results['reasons'] = results['risk_factors']
    
    # Cap risk score
    results['risk_score'] = min(100, max(0, results['risk_score']))
    
    # Generate recommendations if Gemini didn't provide them
    if not results['recommendations']:
        results['recommendations'] = get_recommendations_for_ip(ip_address, results['risk_score'], results['risk_factors'])
    
    # Add severity reason
    results['severity_reason'] = get_severity_reason(results['risk_score'], results['risk_factors'])
    
    # Add confidence explanation
    results['confidence_explanation'] = get_confidence_explanation(results['threat_intel'])
    
    # Determine risk level
    if results['risk_score'] >= 70:
        results['risk_level'] = 'Critical'
        results['risk_color'] = '#dc3545'
    elif results['risk_score'] >= 40:
        results['risk_level'] = 'High'
        results['risk_color'] = '#fd7e14'
    elif results['risk_score'] >= 20:
        results['risk_level'] = 'Medium'
        results['risk_color'] = '#ffc107'
    else:
        results['risk_level'] = 'Low'
        results['risk_color'] = '#28a745'
    
    # Remove duplicates
    results['reasons'] = list(dict.fromkeys(results['reasons']))
    results['recommendations'] = list(dict.fromkeys(results['recommendations']))
    
    return results


def get_recommendations_for_ip(ip_address, risk_score, risk_factors):
    """Generate appropriate recommendations based on risk score and factors"""
    recommendations = []
    
    if risk_score >= 70:
        recommendations.append("🚨 IMMEDIATE ACTION REQUIRED:")
        recommendations.append("  • Block this IP address at firewall level immediately")
        recommendations.append("  • Review logs for any past connections from this IP")
        recommendations.append("  • Scan internal systems for potential compromise")
        recommendations.append("  • Report to your security team/SOC")
        
        if any("C2" in str(factor) for factor in risk_factors):
            recommendations.append("  • Check for beaconing activity in network logs")
        if any("malware" in str(factor).lower() for factor in risk_factors):
            recommendations.append("  • Run antivirus scans on systems that connected to this IP")
            
    elif risk_score >= 40:
        recommendations.append("⚠️ ACTION RECOMMENDED:")
        recommendations.append("  • Block this IP at firewall level")
        recommendations.append("  • Monitor network traffic to/from this IP for 24 hours")
        recommendations.append("  • Add to watchlist for future monitoring")
        
        if any("phishing" in str(factor).lower() for factor in risk_factors):
            recommendations.append("  • Check if any users received emails from this IP")
        if any("scan" in str(factor).lower() for factor in risk_factors):
            recommendations.append("  • Review firewall logs for port scanning attempts")
            
    elif risk_score >= 20:
        recommendations.append("📋 REVIEW RECOMMENDED:")
        recommendations.append("  • Add to monitoring list for 7 days")
        recommendations.append("  • Review logs for any suspicious activity")
        recommendations.append("  • Consider rate-limiting connections from this IP")
    else:
        recommendations.append("✅ No action required - IP appears safe")
    
    # Add specific recommendations based on threat type
    threat_text = ' '.join(str(f).lower() for f in risk_factors)
    
    if "c2" in threat_text or "command and control" in threat_text:
        recommendations.append("  • C2 detected: Check for DNS tunneling and unusual outbound traffic")
    if "tor" in threat_text:
        recommendations.append("  • Tor exit node: Consider blocking if not needed for business")
    if "vpn" in threat_text or "proxy" in threat_text:
        recommendations.append("  • VPN/Proxy: Verify if legitimate business use")
    if "ddos" in threat_text:
        recommendations.append("  • DDoS risk: Ensure rate limiting and DDoS protection is enabled")
    if "phishing" in threat_text:
        recommendations.append("  • Phishing host: Alert users about potential phishing emails")
    if "scan" in threat_text:
        recommendations.append("  • Scanning activity: Harden exposed services and update firewall rules")
    
    return list(dict.fromkeys(recommendations))


def get_severity_reason(risk_score, risk_factors):
    """Generate a clear reason for the severity level"""
    if risk_score >= 70:
        return "CRITICAL: Immediate threat detected - take action now"
    elif risk_score >= 40:
        return "HIGH: Significant risk detected - investigate and consider blocking"
    elif risk_score >= 20:
        return "MEDIUM: Potential risk - monitor activity"
    else:
        return "LOW: Minimal risk - no immediate action needed"


def get_confidence_explanation(threat_intel):
    """Explain why AbuseIPDB confidence is low"""
    explanation = []
    
    if threat_intel.get('abuseipdb', {}).get('confidence', 0) == 0:
        explanation.append("• This IP is not in AbuseIPDB's database (no reports from security researchers)")
        explanation.append("• This doesn't mean the IP is safe - just that it hasn't been reported yet")
        explanation.append("• Detection based on other threat intelligence sources and behavior analysis")
    
    if threat_intel.get('abuseipdb', {}).get('total_reports', 0) == 0:
        explanation.append("• No prior abuse reports filed against this IP")
    
    return explanation


# Route to serve images from 5G_IDS_Results folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_PATH, filename)


# Feature descriptions
feature_descriptions = {
    'Offset': 'Packet offset in network flow',
    'sMeanPktSz': 'Source mean packet size',
    'sTtl': 'Source Time To Live',
    'sHops': 'Source hops count',
    'TcpRtt': 'TCP Round Trip Time',
    'State': 'Connection state',
    'AckDat': 'ACK data packets',
    'SynAck': 'SYN-ACK packets',
    'Proto': 'Protocol (1=TCP, 2=UDP, 3=ICMP)',
    'SrcBytes': 'Source bytes transmitted',
    'TotBytes': 'Total bytes transmitted',
    'SrcRate': 'Source transmission rate',
    'Max': 'Maximum packet size',
    'Load': 'Network load',
    'Dur': 'Flow duration',
    'SrcLoad': 'Source load',
    'Rate': 'Transmission rate',
    'RunTime': 'Runtime of the flow',
    'Cause': 'Cause of connection termination',
    'Min': 'Minimum packet size'
}

# Pre-defined examples
EXAMPLES = {
    'normal_1': {
        'name': 'Normal Traffic - Web Browsing',
        'values': [175604.0, 245.5, 64.0, 0.0, 0.0, 5.0, 100.0, 50.0, 1.0, 50000.0, 100000.0, 1000.0, 1500.0, 0.5, 10.0, 0.5, 1000.0, 10.0, 0.0, 64.0]
    },
    'normal_2': {
        'name': 'Normal Traffic - Video Streaming',
        'values': [156234.0, 312.8, 128.0, 1.0, 0.0, 4.0, 120.0, 60.0, 1.0, 45000.0, 95000.0, 900.0, 1450.0, 0.4, 12.0, 0.4, 900.0, 12.0, 0.0, 62.0]
    },
    'attack_1': {
        'name': 'Attack - UDP Flood',
        'values': [19272380.0, 42.0, 63.0, 1.0, 0.0, 8.0, 10.0, 5.0, 2.0, 10000000.0, 20000000.0, 50000.0, 1500.0, 0.9, 2.0, 0.9, 50000.0, 2.0, 2.0, 64.0]
    },
    'attack_2': {
        'name': 'Attack - SYN Flood',
        'values': [24567890.0, 38.0, 32.0, 2.0, 0.0, 9.0, 5.0, 2.0, 1.0, 15000000.0, 30000000.0, 60000.0, 1480.0, 0.95, 1.5, 0.95, 60000.0, 1.5, 2.0, 60.0]
    }
}

NETWORK_PROFILES = {
    '4g': {
        'label': '4G LTE IDS Mode',
        'core': 'EPC',
        'radio': 'eNodeB',
        'latency_ms': '35-60',
        'throughput_mbps': '75-150',
        'control_plane': 'MME / S-GW / P-GW',
        'ids_focus': [
            'S1-U and SGi traffic inspection',
            'GTP tunnel anomaly detection',
            'SYN/UDP flood detection at EPC edge',
            'IP reputation checks for internet breakouts'
        ],
        'normal_baseline': {
            'latency': 48,
            'packet_rate': 900,
            'load': 42,
            'slice_risk': 18
        }
    },
    '5g': {
        'label': '5G Standalone IDS Mode',
        'core': '5GC',
        'radio': 'gNodeB',
        'latency_ms': '5-20',
        'throughput_mbps': '500-1000+',
        'control_plane': 'AMF / SMF / UPF',
        'ids_focus': [
            'N3/N6 user-plane monitoring',
            'UPF traffic behavior profiling',
            'Network slice isolation alerts',
            'Low-latency flow anomaly detection'
        ],
        'normal_baseline': {
            'latency': 12,
            'packet_rate': 3200,
            'load': 58,
            'slice_risk': 26
        }
    }
}

ATTACK_CATEGORIES = {
    'udp_flood': {
        'label': 'UDP Flood',
        'family': 'Volumetric DoS',
        'description': 'High-rate UDP traffic that can exhaust radio/core bandwidth and IDS processing capacity.',
        '4g_surface': 'EPC / S1-U bearer flooding',
        '5g_surface': 'UPF / N3 user-plane flooding',
        'mitigation': 'Rate-limit UDP bursts, apply bearer/session throttling, and block abusive sources.'
    },
    'syn_flood': {
        'label': 'TCP SYN Flood',
        'family': 'Protocol DoS',
        'description': 'Large number of incomplete TCP handshakes intended to exhaust connection state.',
        '4g_surface': 'SGi internet edge and EPC gateway state pressure',
        '5g_surface': 'N6 breakout and UPF session state pressure',
        'mitigation': 'Enable SYN cookies, tune connection limits, and isolate affected gateway path.'
    },
    'gtp_tunnel_abuse': {
        'label': 'GTP Tunnel Abuse',
        'family': 'Mobile Core Abuse',
        'description': 'Suspicious mobile-core tunnel behavior, often visible as abnormal bearer volume or rate.',
        '4g_surface': 'GTP-U tunnel across S1-U / S5 / S8',
        '5g_surface': 'GTP-U tunnel across N3 / N9',
        'mitigation': 'Inspect tunnel endpoint pairs, validate TEID mappings, and quarantine suspicious sessions.'
    },
    'scanning_probe': {
        'label': 'Reconnaissance / Scan',
        'family': 'Reconnaissance',
        'description': 'Low-to-medium duration probing behavior that may precede exploitation.',
        '4g_surface': 'SGi exposed service probing',
        '5g_surface': 'N6 service probing and edge workload discovery',
        'mitigation': 'Throttle repeated probes, enrich source reputation, and tighten exposed service rules.'
    },
    'slice_anomaly': {
        'label': '5G Slice Anomaly',
        'family': 'Slice Isolation Risk',
        'description': '5G-like high-load/low-latency flow that may indicate slice resource abuse.',
        '4g_surface': 'Not applicable to LTE slicing; treat as EPC QoS anomaly',
        '5g_surface': 'Network slice / UPF policy drift',
        'mitigation': 'Validate slice policy, check QoS flows, and enforce per-slice rate and isolation controls.'
    },
    'generic_anomaly': {
        'label': 'Generic Traffic Anomaly',
        'family': 'Behavioral Anomaly',
        'description': 'The IDS model detected suspicious traffic but features do not map cleanly to one attack family.',
        '4g_surface': 'EPC traffic anomaly',
        '5g_surface': '5GC traffic anomaly',
        'mitigation': 'Escalate for analyst review and correlate with recent IP reputation and flow logs.'
    },
    'normal': {
        'label': 'Normal / Benign',
        'family': 'Baseline',
        'description': 'Traffic is currently inside the model baseline.',
        '4g_surface': 'Expected LTE bearer behavior',
        '5g_surface': 'Expected 5G user-plane behavior',
        'mitigation': 'Continue monitoring.'
    }
}


def _safe_float(value, default=0.0):
    """Convert flow values safely for IDS context scoring."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_network_generation(values):
    """Infer whether a traffic sample behaves more like 4G LTE or 5G SA."""
    if len(values) != 20:
        return {
            'inferred_network': 'unknown',
            'confidence': 0,
            'reason': 'Expected 20 IDS feature values for network generation analysis.'
        }

    tcp_rtt = _safe_float(values[4])
    rate = _safe_float(values[16])
    src_rate = _safe_float(values[11])
    load = _safe_float(values[13])
    duration = _safe_float(values[14])
    total_bytes = _safe_float(values[10])
    packet_size = _safe_float(values[1])

    score_5g = 0
    score_4g = 0
    reasons = []

    if tcp_rtt and tcp_rtt <= 20:
        score_5g += 2
        reasons.append('low RTT fits 5G user-plane behavior')
    elif tcp_rtt >= 35:
        score_4g += 2
        reasons.append('higher RTT fits 4G EPC behavior')

    if rate >= 2500 or src_rate >= 2500:
        score_5g += 2
        reasons.append('high packet/flow rate is closer to 5G throughput')
    elif rate < 1800 and src_rate < 1800:
        score_4g += 1
        reasons.append('moderate traffic rate is closer to 4G LTE')

    if total_bytes >= 250000:
        score_5g += 1
        reasons.append('large flow volume suggests broadband 5G capacity')
    else:
        score_4g += 1

    if duration <= 5 and load >= 0.6:
        score_5g += 1
        reasons.append('short high-load burst resembles low-latency 5G traffic')
    elif duration >= 8:
        score_4g += 1

    if packet_size >= 300:
        score_5g += 1
    else:
        score_4g += 1

    inferred = '5g' if score_5g >= score_4g else '4g'
    confidence = int((max(score_4g, score_5g) / max(1, score_4g + score_5g)) * 100)

    return {
        'inferred_network': inferred,
        'confidence': confidence,
        'reason': '; '.join(reasons[:4]) or 'Traffic feature mix is close to the selected network baseline.'
    }


def build_network_context(values, selected_network='5g', probability=None):
    """Build additive 4G/5G IDS context without changing the ML prediction."""
    selected_network = selected_network if selected_network in NETWORK_PROFILES else '5g'
    profile = NETWORK_PROFILES[selected_network]
    inferred = _classify_network_generation(values)
    attack_probability = float(probability) if probability is not None else float(predict(values))

    src_rate = _safe_float(values[11])
    rate = _safe_float(values[16])
    load = _safe_float(values[13])
    tcp_rtt = _safe_float(values[4])
    duration = _safe_float(values[14])
    proto = int(_safe_float(values[8]))
    total_bytes = _safe_float(values[10])

    baseline = profile['normal_baseline']
    packet_rate = int(max(src_rate, rate, baseline['packet_rate']))
    load_percent = int(min(100, max(0, load * 100 if load <= 1.5 else load)))
    latency = int(tcp_rtt if tcp_rtt > 0 else baseline['latency'])
    anomaly_pressure = int(min(100, max(0, attack_probability * 100)))
    slice_risk = int(min(100, baseline['slice_risk'] + (anomaly_pressure * 0.45) + (load_percent * 0.15)))
    attack_category = categorize_attack(values, selected_network, attack_probability)

    ids_path = [
        profile['radio'],
        'Transport',
        profile['core'],
        'IDS Sensor',
        'AI Classifier',
        'SOC Response'
    ]

    if selected_network == '4g':
        threat_surface = [
            'S1-U GTP-U tunnel abuse',
            'EPC edge volumetric flood',
            'SGi internet breakout reputation risk',
            'Legacy control-plane signaling spikes'
        ]
    else:
        threat_surface = [
            'N3 user-plane anomaly',
            'UPF breakout abuse',
            'Slice isolation drift',
            'Low-latency burst attack'
        ]

    return {
        'selected_network': selected_network,
        'profile': profile,
        'inference': inferred,
        'matches_selected': inferred['inferred_network'] == selected_network,
        'metrics': {
            'latency_ms': latency,
            'packet_rate': packet_rate,
            'load_percent': load_percent,
            'flow_duration': duration,
            'total_bytes': int(total_bytes),
            'attack_probability': round(anomaly_pressure, 2),
            'slice_risk': slice_risk,
            'protocol': {1: 'TCP', 2: 'UDP', 3: 'ICMP'}.get(proto, f'PROTO-{proto}')
        },
        'ids_path': ids_path,
        'threat_surface': threat_surface,
        'attack_category': attack_category,
        'recommendation': (
            attack_category['mitigation']
            if attack_probability > 0.5 else
            'Continue monitoring; current traffic is within the selected network baseline.'
        )
    }


@lru_cache(maxsize=128)
def cached_network_context(values_tuple, selected_network='5g', probability=None):
    """Cache repeated demo/example context calculations for a smoother dashboard."""
    values = list(values_tuple)
    return build_network_context(values, selected_network, probability)


def categorize_attack(values, network_type='5g', probability=0.0):
    """Classify likely attack family from IDS flow features and selected mobile network."""
    if len(values) != 20:
        category = ATTACK_CATEGORIES['generic_anomaly']
        return {
            'key': 'generic_anomaly',
            **category,
            'confidence': 40,
            'evidence': ['Feature vector length is invalid, so category is uncertain.'],
            'network_surface': category.get(f'{network_type}_surface', category['5g_surface'])
        }

    proto = int(_safe_float(values[8]))
    src_bytes = _safe_float(values[9])
    total_bytes = _safe_float(values[10])
    src_rate = _safe_float(values[11])
    mean_packet = _safe_float(values[1])
    ack_dat = _safe_float(values[6])
    syn_ack = _safe_float(values[7])
    load = _safe_float(values[13])
    duration = _safe_float(values[14])
    rate = _safe_float(values[16])
    cause = _safe_float(values[18])
    tcp_rtt = _safe_float(values[4])
    attack_probability = float(probability or 0.0)

    if attack_probability <= 0.5:
        category = ATTACK_CATEGORIES['normal']
        return {
            'key': 'normal',
            **category,
            'confidence': int(max(55, (1 - attack_probability) * 100)),
            'evidence': ['Binary IDS model classified this traffic as normal.'],
            'network_surface': category.get(f'{network_type}_surface', category['5g_surface'])
        }

    scores = {
        'udp_flood': 0,
        'syn_flood': 0,
        'gtp_tunnel_abuse': 0,
        'scanning_probe': 0,
        'slice_anomaly': 0,
        'generic_anomaly': 1
    }
    evidence = {key: [] for key in scores}

    if proto == 2:
        scores['udp_flood'] += 35
        evidence['udp_flood'].append('Protocol is UDP.')
    if src_rate >= 25000 or rate >= 25000:
        scores['udp_flood'] += 25
        evidence['udp_flood'].append('Extremely high source/flow rate.')
    if total_bytes >= 5000000:
        scores['udp_flood'] += 20
        evidence['udp_flood'].append('Large byte volume in one flow.')
    if mean_packet <= 80 and duration <= 3:
        scores['udp_flood'] += 10
        evidence['udp_flood'].append('Small-packet short burst pattern.')

    if proto == 1:
        scores['syn_flood'] += 20
        evidence['syn_flood'].append('Protocol is TCP.')
    if syn_ack <= 10 and ack_dat <= 20 and rate >= 10000:
        scores['syn_flood'] += 35
        evidence['syn_flood'].append('Low ACK/SYN-ACK completion with high rate.')
    if duration <= 3 and load >= 0.8:
        scores['syn_flood'] += 15
        evidence['syn_flood'].append('Short high-load connection burst.')
    if cause >= 2:
        scores['syn_flood'] += 10
        evidence['syn_flood'].append('Abnormal termination cause indicator.')

    if total_bytes >= 1000000 and max(src_rate, rate) >= 5000:
        scores['gtp_tunnel_abuse'] += 20
        evidence['gtp_tunnel_abuse'].append('High mobile-core tunnel volume/rate.')
    if network_type == '4g' and duration <= 5 and load >= 0.6:
        scores['gtp_tunnel_abuse'] += 20
        evidence['gtp_tunnel_abuse'].append('4G EPC burst resembles bearer tunnel abuse.')
    if network_type == '5g' and total_bytes >= 1000000:
        scores['gtp_tunnel_abuse'] += 12
        evidence['gtp_tunnel_abuse'].append('5G N3/N9 tunnel can carry large abnormal bursts.')

    if 300 <= max(src_rate, rate) <= 5000 and duration >= 8:
        scores['scanning_probe'] += 25
        evidence['scanning_probe'].append('Moderate rate over longer duration.')
    if src_bytes < 100000 and mean_packet < 180:
        scores['scanning_probe'] += 15
        evidence['scanning_probe'].append('Small request-like traffic pattern.')
    if proto in {1, 3} and tcp_rtt >= 20:
        scores['scanning_probe'] += 10
        evidence['scanning_probe'].append('Probe-like TCP/ICMP behavior.')

    if network_type == '5g':
        if tcp_rtt <= 20 and max(src_rate, rate) >= 2500:
            scores['slice_anomaly'] += 25
            evidence['slice_anomaly'].append('Low-latency high-rate 5G flow.')
        if load >= 0.8 and duration <= 5:
            scores['slice_anomaly'] += 20
            evidence['slice_anomaly'].append('High load burst can affect slice resource isolation.')

    best_key = max(scores, key=scores.get)
    if scores[best_key] < 25:
        best_key = 'generic_anomaly'
        evidence[best_key].append('Suspicious model score without a strong signature match.')

    category = ATTACK_CATEGORIES[best_key]
    confidence = int(min(98, max(50, scores[best_key] + (attack_probability * 25))))
    return {
        'key': best_key,
        **category,
        'confidence': confidence,
        'evidence': evidence[best_key][:4],
        'network_surface': category.get(f'{network_type}_surface', category['5g_surface'])
    }


def _severity_from_probability(probability):
    """Map model probability to SOC-style alert severity."""
    if probability >= 0.85:
        return 'Critical'
    if probability >= 0.65:
        return 'High'
    if probability >= 0.5:
        return 'Medium'
    return 'Informational'


def _recommended_response(severity, network_type, attack_category=None):
    """Generate fast response guidance for dashboard alerts."""
    if attack_category and attack_category in ATTACK_CATEGORIES and attack_category != 'normal':
        return ATTACK_CATEGORIES[attack_category]['mitigation']
    if severity == 'Critical':
        return f'Isolate affected {network_type.upper()} path, block source, and escalate to SOC.'
    if severity == 'High':
        return f'Apply rate limits and inspect recent {network_type.upper()} flows.'
    if severity == 'Medium':
        return f'Add {network_type.upper()} flow to watchlist and increase sampling.'
    return 'No immediate action required; continue baseline monitoring.'


def _add_alert(source, title, severity, network_type='5g', score=0, details=None, attack_category=None):
    """Store a bounded in-memory alert for the live IDS console."""
    details = details or {}
    category_key = attack_category or details.get('attack_category') or 'normal'
    category = ATTACK_CATEGORIES.get(category_key, ATTACK_CATEGORIES['generic_anomaly'])
    alert = {
        'id': f'AEGIS-{datetime.utcnow().strftime("%Y%m%d%H%M%S%f")}',
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'source': source,
        'title': title,
        'severity': severity,
        'network_type': network_type,
        'attack_category': category_key,
        'attack_type': category['label'],
        'attack_family': category['family'],
        'score': round(float(score), 2),
        'details': details,
        'recommended_response': _recommended_response(severity, network_type, category_key)
    }
    ALERT_HISTORY.appendleft(alert)
    return alert


def _alert_stats():
    """Summarize current in-memory IDS alert state."""
    total = len(ALERT_HISTORY)
    attacks = sum(1 for alert in ALERT_HISTORY if alert['severity'] in {'Critical', 'High', 'Medium'})
    critical = sum(1 for alert in ALERT_HISTORY if alert['severity'] == 'Critical')
    high = sum(1 for alert in ALERT_HISTORY if alert['severity'] == 'High')
    by_network = {
        '4g': sum(1 for alert in ALERT_HISTORY if alert.get('network_type') == '4g'),
        '5g': sum(1 for alert in ALERT_HISTORY if alert.get('network_type') == '5g')
    }
    by_attack_type = {}
    for alert in ALERT_HISTORY:
        attack_type = alert.get('attack_type', 'Unknown')
        by_attack_type[attack_type] = by_attack_type.get(attack_type, 0) + 1
    return {
        'total_alerts': total,
        'attack_alerts': attacks,
        'critical_alerts': critical,
        'high_alerts': high,
        'normal_events': max(0, total - attacks),
        'by_network': by_network,
        'by_attack_type': by_attack_type,
        'history_limit': MAX_ALERT_HISTORY
    }


def predict(values):
    """Make prediction"""
    input_array = np.array(values, dtype=np.float32).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return probability


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         features=feature_names, 
                         descriptions=feature_descriptions,
                         examples=EXAMPLES)


@app.route('/predict', methods=['POST'])
def predict_api():
    """Prediction API endpoint"""
    try:
        data = request.get_json()
        
        if 'values' in data:
            values = data['values']
        elif 'csv_values' in data:
            csv_str = data['csv_values'].strip()
            values = [float(x.strip()) for x in csv_str.split(',')]
        else:
            return jsonify({'error': 'No values provided'}), 400
        
        if len(values) != 20:
            return jsonify({'error': f'Expected 20 values, got {len(values)}'}), 400
        
        probability = predict(values)
        is_attack = probability > 0.5
        requested_network = data.get('network_type', '5g')
        network_type = requested_network if requested_network in NETWORK_PROFILES else '5g'
        severity = _severity_from_probability(float(probability))
        attack_category = categorize_attack(values, network_type, float(probability))
        
        result = {
            'is_attack': bool(is_attack),
            'probability': float(probability),
            'confidence': float(probability * 100 if is_attack else (1 - probability) * 100),
            'prediction': 'ATTACK' if is_attack else 'NORMAL',
            'threat_level': 'HIGH' if probability > 0.8 else 'MEDIUM' if probability > 0.6 else 'LOW' if is_attack else 'SAFE',
            'severity': severity,
            'network_type': network_type,
            'attack_category': attack_category
        }

        alert_title = f"{attack_category['label']} detected" if is_attack else 'Normal traffic baseline event'
        _add_alert(
            source='traffic_model',
            title=alert_title,
            severity=severity,
            network_type=network_type,
            score=float(probability) * 100,
            attack_category=attack_category['key'],
            details={
                'prediction': result['prediction'],
                'threat_level': result['threat_level'],
                'confidence': result['confidence'],
                'attack_category': attack_category['key'],
                'attack_type': attack_category['label'],
                'attack_family': attack_category['family'],
                'category_confidence': attack_category['confidence'],
                'evidence': attack_category['evidence'],
                'network_surface': attack_category['network_surface']
            }
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_ip', methods=['POST'])
def check_ip():
    """API endpoint to check if an IP is malicious"""
    data = request.get_json()
    ip_address = data.get('ip', '').strip()
    
    if not ip_address:
        return jsonify({'error': 'IP address required'}), 400
    
    # Validate IP format
    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        return jsonify({'error': 'Invalid IP address format'}), 400
    
    # Run reputation check
    result = check_ip_reputation(ip_address)
    severity = result.get('risk_level', 'Low')
    normalized_severity = 'Informational' if severity == 'Low' else severity
    _add_alert(
        source='ip_reputation',
        title=f'IP reputation checked: {ip_address}',
        severity=normalized_severity,
        network_type=data.get('network_type', '5g') if data.get('network_type') in NETWORK_PROFILES else '5g',
        score=result.get('risk_score', 0),
        details={
            'ip': ip_address,
            'risk_level': result.get('risk_level'),
            'risk_score': result.get('risk_score'),
            'risk_factors': result.get('risk_factors', [])[:5]
        }
    )
    
    return jsonify(result)


@app.route('/network_profiles', methods=['GET'])
def network_profiles():
    """Return 4G and 5G IDS profile data for charts and diagrams."""
    return jsonify(NETWORK_PROFILES)


@app.route('/attack_categories', methods=['GET'])
def attack_categories():
    """Return attack categories supported by the IDS explanation layer."""
    return jsonify(ATTACK_CATEGORIES)


@app.route('/network_context', methods=['POST'])
def network_context():
    """Analyze IDS traffic as either 4G LTE or 5G SA without changing /predict."""
    try:
        data = request.get_json()
        selected_network = data.get('network_type', '5g')

        if 'values' in data:
            values = [float(x) for x in data['values']]
        elif 'csv_values' in data:
            csv_str = data['csv_values'].strip()
            values = [float(x.strip()) for x in csv_str.split(',')]
        else:
            return jsonify({'error': 'No values provided'}), 400

        if len(values) != 20:
            return jsonify({'error': f'Expected 20 values, got {len(values)}'}), 400

        probability = data.get('probability')
        values_tuple = tuple(round(float(v), 6) for v in values)
        context = cached_network_context(values_tuple, selected_network, probability)
        return jsonify(context)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/alerts', methods=['GET'])
def alerts():
    """Return recent IDS alerts for the dashboard SOC console."""
    severity = request.args.get('severity', '').strip()
    network_type = request.args.get('network_type', '').strip()
    attack_category = request.args.get('attack_category', '').strip()
    limit = min(int(request.args.get('limit', 25)), MAX_ALERT_HISTORY)

    alerts_list = list(ALERT_HISTORY)
    if severity:
        alerts_list = [alert for alert in alerts_list if alert['severity'].lower() == severity.lower()]
    if network_type in NETWORK_PROFILES:
        alerts_list = [alert for alert in alerts_list if alert.get('network_type') == network_type]
    if attack_category in ATTACK_CATEGORIES:
        alerts_list = [alert for alert in alerts_list if alert.get('attack_category') == attack_category]

    return jsonify({
        'alerts': alerts_list[:limit],
        'stats': _alert_stats()
    })


@app.route('/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear in-memory dashboard alerts."""
    ALERT_HISTORY.clear()
    return jsonify({'status': 'cleared', 'stats': _alert_stats()})


if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print(f"📍 Open http://127.0.0.1:5000 in your browser")
    print("="*50)
    app.run(debug=True, host='127.0.0.1', port=5000)
