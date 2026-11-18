"""User-reviewed image quality exclusions, independent of the host filesystem."""
import hashlib,json
from pathlib import Path,PureWindowsPath
POLICY=Path(__file__).resolve().parent/'quality_exclusions.json'

def policy_hash():
    return hashlib.sha256(POLICY.read_bytes()).hexdigest()

def image_key(path):
    parts=PureWindowsPath(str(path)).parts
    return '/'.join(parts[-2:])

def load_policy():
    document=json.loads(POLICY.read_text(encoding='utf-8'))
    result={}
    for group in document['groups']:
        for name in group['filenames']:
            key=group['event_folder']+'/'+name
            if key in result: raise ValueError(f'Duplicate exclusion: {key}')
            result[key]=group['reason']
    return result

def filter_quality(rows,negative_rows):
    policy=load_policy()
    positive={image_key(r['original_image_path']):r for r in rows}
    negative={image_key(r['original_image_path']):r for r in negative_rows}
    missing=set(policy)-set(positive)-set(negative)
    if missing: raise ValueError(f'Quality exclusions absent from source manifest: {sorted(missing)}')
    kept=[r for r in rows if image_key(r['original_image_path']) not in policy]
    report=[]
    for key,reason in policy.items():
        row=(positive if key in positive else negative)[key]
        report.append({**row,'reason':reason,'status':'excluded_quality' if key in positive else 'already_excluded_negative'})
    return kept,report

def assert_quality_compliant(rows,metadata):
    policy=load_policy()
    bad=[r['original_image_path'] for r in rows if image_key(r['original_image_path']) in policy]
    if bad: raise ValueError(f'Training manifest contains {len(bad)} reviewed quality exclusions')
    if metadata.get('quality_policy_sha256')!=policy_hash():
        raise ValueError('Dataset quality policy is stale; refresh the filtered manifest before training')
