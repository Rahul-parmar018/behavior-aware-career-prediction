"""
Maps AMEO free-text job titles (Designation, 419 unique values) to career domains.

Rules are ordered: the first matching pattern wins. Titles that do not identify
a domain (e.g. "engineer", "project engineer", "graduate engineer trainee") and
academic / research roles (too few to form a class) return None and are excluded
from classification. Note: at Indian IT service firms "system(s) engineer" and
"programmer analyst" are entry-level software-development titles, so they are
mapped to Software_Dev.
"""
import re

AMBIGUOUS = {
    "engineer", "senior engineer", "project engineer", "junior engineer",
    "trainee engineer", "graduate engineer trainee", "get", "associate engineer",
    "technical engineer", "application engineer", "product engineer",
    "executive engineer", "lead engineer", "assistant engineer", "sr. engineer",
    "graduate engineer", "graduate trainee engineer", "engineer trainee",
    "apprentice", "graduate apprentice trainee", "it engineer", "technical assistant",
    "technical consultant", "ase", "executive engg", "product development engineer",
    "research engineer", "r&d engineer", "r & d", "engineer-hws", "shift engineer",
}

RULES = [
    ("Academic", r"lecturer|professor|faculty|educator|research fellow|research associate|"
                 r"research scientist|research staff"),
    ("Core_Engineering", r"quality control|electrical|mechanical|civil|production|manufactur|"
                         r"maintenance|design engineer|designer engineer|hardware|embedded|firmware|"
                         r"\brf\b|rf/|telecom|electronic|field engineer|field service|site engineer|"
                         r"site manager|cad|instrumentation|control|dcs|planning engineer|"
                         r"process engineer|industrial|environmental|aircraft|cnc|bss|"
                         r"continuous improvement|engineering technician|electrical designer"),
    ("QA_Testing", r"test|qa\b|qa |quality assurance|quality analyst|quality engineer|"
                   r"software quality|quality associate|quality consultant|performance engineer|"
                   r"automation engineer"),
    ("Data_Business_Analysis", r"data analyst|data scientist|decision scientist|business analyst|"
                               r"business system|business intelligence|research analyst|"
                               r"marketing analyst|financial analyst|operations analyst|"
                               r"business process analyst|business technology analyst|"
                               r"supply chain analyst|portfolio analyst|risk|\bmis\b|"
                               r"business consultant|it business analyst"),
    ("IT_Infra_Support", r"network|system administrator|systems administrator|administrator|"
                         r"\bdba\b|database administrator|technical support|it support|"
                         r"desktop support|help desk|it technician|information security|"
                         r"security analyst|linux|cloud|noc|it executive|support engineer|"
                         r"customer support engineer|it operations|technical operations|"
                         r"database engineer|implementation engineer|information technology specialist|"
                         r"it assistant|product support|telecom support"),
    ("Software_Dev", r"software|developer|programmer|program analyst|programmer analyst|"
                     r"web|android|ios|\.net|dotnet|java|php|sap|salesforce|full stack|"
                     r"ui developer|game|etl|technical lead|technology lead|team lead|"
                     r"architect|system engineer|systems engineer|system analyst|systems analyst|"
                     r"it analyst|technology analyst|technical analyst|sharepoint|teradata|"
                     r"python|ruby|javascript|html|c# |c#$|enterprise solutions|software engg"),
    ("Management_Sales_Ops", r"manager|management|sales|business development|marketing|"
                             r"customer|client|operation|\bhr\b|human resource|recruit|talent|"
                             r"staffing|process associate|process executive|process advisor|"
                             r"account|seo|data entry|admin|clerical|office|secretary|"
                             r"executive assistant|coordinator|logistics|catalog|banking|"
                             r"training|trainer|consultant|editor|writer|documentation|"
                             r"loss prevention|investigator|service|associate|executive|"
                             r"project assistant|assistant"),
]
_COMPILED = [(name, re.compile(p)) for name, p in RULES]

DOMAINS = ["Software_Dev", "QA_Testing", "IT_Infra_Support",
           "Data_Business_Analysis", "Core_Engineering", "Management_Sales_Ops"]


def map_designation(title):
    t = str(title).lower().strip()
    if t in AMBIGUOUS:
        return None
    for name, pat in _COMPILED:
        if pat.search(t):
            return name if name in DOMAINS else None
    return None
