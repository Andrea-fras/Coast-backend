"""Sanitized notes and optimistic concurrency, shared by both editors."""
import hashlib
import bleach
from bleach.css_sanitizer import CSSSanitizer

TAGS = ['p','br','div','span','b','strong','i','em','u','s','h1','h2','h3','h4',
        'ul','ol','li','blockquote','pre','code','table','thead','tbody','tr','th','td',
        'hr','mark','font','a','img','sub','sup']
CSS = CSSSanitizer(allowed_css_properties=['color','background-color','font-size',
    'font-weight','font-style','text-decoration','text-align'])
def sanitize_notes(html):
    return bleach.clean(str(html or ''), tags=TAGS, attributes={
        '*': ['class','style','title'], 'a': ['href'], 'img': ['src','alt'],
        'font': ['color','size'], 'td': ['colspan','rowspan'], 'th': ['colspan','rowspan'],
    }, protocols=['http','https','mailto','data'], css_sanitizer=CSS, strip=True)
def notes_revision(html):
    return hashlib.sha256(str(html or '').encode('utf-8')).hexdigest()
