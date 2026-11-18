"""UT labels and timestamp masking. Originals remain unchanged."""
import argparse,csv,hashlib,json,re
from datetime import datetime
from pathlib import Path
import numpy as np
import openpyxl
from PIL import Image,ImageDraw
from quality_filter import filter_quality,policy_hash
BASE=Path(__file__).resolve().parent
FIELDS=['event_id','original_image_path','processed_image_path','image_time_UT','CME_time_UT','arrival_time_UT','transit_time_hours']
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(path,rows,fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def load_events(excel):
    workbook=openpyxl.load_workbook(excel,read_only=True,data_only=True)
    try:
        rows=iter(workbook['Final_trainable_148'].values); headers=next(rows); result={}
        for values in rows:
            if values[0] is None: continue
            row=dict(zip(headers,values)); eid=int(row['event_id'])
            if eid in result: raise ValueError(f'Duplicate event {eid}')
            for key in ('CME_time_UT','arrival_time_UT'): row[key]=datetime.fromisoformat(str(row[key]))
            result[eid]=row
        return result
    finally: workbook.close()
def build_manifest(root,excel):
    events=load_events(excel); accepted=[]; excluded=[]; notes=[]; seen=set()
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir(): continue
        match=re.fullmatch(r'event_(\d+)_(\d{8})_(\d{6})',folder.name)
        if not match: raise ValueError(f'Unknown event directory: {folder}')
        eid=int(match[1])
        if eid not in events or eid in seen: raise ValueError(f'Unknown/duplicate event {eid}')
        seen.add(eid); event=events[eid]
        start=datetime.strptime(match[2]+match[3],'%Y%m%d%H%M%S')
        offset=(start-event['CME_time_UT']).total_seconds()
        if abs(offset)>1: raise ValueError(f'Unreviewed CME time offset {eid}: {offset}s')
        if offset: notes.append({'event_id':eid,'accepted_CME_time_offset_seconds':offset})
        records=read_csv(folder/'images.csv'); metadata={r['保存文件名']:r for r in records}
        paths=sorted(folder.glob('*.png'))
        if len(metadata)!=len(records) or not paths or set(metadata)!={p.name for p in paths}:
            raise ValueError(f'Missing/duplicate/unlisted image records: {folder}')
        times=set()
        for path in paths:
            match=re.fullmatch(r'\d+_(\d{8})_(\d{6})\.png',path.name)
            if not match: raise ValueError(f'Unknown filename: {path}')
            time=datetime.strptime(match[1]+match[2],'%Y%m%d%H%M%S'); record=metadata[path.name]
            if int(record['event编号'])!=eid or datetime.fromisoformat(record['Image_Time'])!=time:
                raise ValueError(f'Manifest mismatch: {path}')
            if time in times or time<event['CME_time_UT']: raise ValueError(f'Duplicate/early image: {path}')
            times.add(time)
            hours=(event['arrival_time_UT']-time).total_seconds()/3600
            row=dict(zip(FIELDS,[eid,str(path.resolve()),'',time.isoformat(' '),event['CME_time_UT'].isoformat(' '),event['arrival_time_UT'].isoformat(' '),hours]))
            (excluded if hours<0 else accepted).append(row)
    if seen!=set(events): raise ValueError(f'Missing event folders: {set(events)-seen}')
    return accepted,excluded,notes

def preprocess(path,mask_top=487):
    with Image.open(path) as source:
        source.load()
        if source.size!=(512,512): raise ValueError(f'Unexpected dimensions: {path}')
        gray=source.convert('L')
    if not 480<=mask_top<=490: raise ValueError('mask_top must be 480..490 for review')
    # Interior of the coronagraph baffle ring; verify uniform before using its color.
    patch=np.asarray(gray)[240:250,180:190]; fill=int(np.median(patch))
    if np.ptp(patch)!=0: raise ValueError(f'Nonuniform baffle sample, inspect {path}')
    masked=gray.copy(); ImageDraw.Draw(masked).rectangle((0,mask_top,511,511),fill=fill)
    return gray,masked,masked.resize((256,256),Image.Resampling.BILINEAR),fill

def choose_samples(rows):
    selected=[]
    for eid in [2,4,14,61,64,94,107,131,152]:
        group=[r for r in rows if int(r['event_id'])==eid]
        if group:
            selected.append(group[-1] if eid in (14,61) else group[0])
            if eid==2 and len(group)>1: selected.append(group[len(group)//2])
    return selected

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(r'D:\cmepicturedownload\LASCO_148'))
    parser.add_argument('--excel',type=Path,default=Path(r'D:\CME\CME_148_final.xlsx'))
    parser.add_argument('--output',type=Path,default=BASE/'outputs/review_samples')
    parser.add_argument('--mode',choices=['sample','batch'],default='sample')
    parser.add_argument('--mask-top',type=int,default=487)
    parser.add_argument('--review-approved',action='store_true')
    args=parser.parse_args()
    if args.mode=='batch' and not args.review_approved: parser.error('Batch requires human sample review and --review-approved')
    out=args.output.resolve(); root=args.root.resolve()
    if out==root or root in out.parents: parser.error('Output must be outside original image tree')
    if out.exists() and any(out.iterdir()): parser.error('Choose a new empty output directory')
    rows,excluded,notes=build_manifest(root,args.excel)
    source_count=len(rows)+len(excluded)
    rows,quality_report=filter_quality(rows,excluded)
    selected=choose_samples(rows) if args.mode=='sample' else rows
    out.mkdir(parents=True,exist_ok=True); labels=[]; checks=[]; panels=[]
    for original in selected:
        row=original.copy(); path=Path(row['original_image_path']); before=hashlib.sha256(path.read_bytes()).hexdigest()
        gray,masked,image,fill=preprocess(path,args.mask_top)
        target=out/'images'/path.parent.name/path.name; target.parent.mkdir(parents=True,exist_ok=True); image.save(target)
        row['processed_image_path']=str(target); labels.append(row)
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        assert np.array_equal(np.asarray(gray)[:args.mask_top],np.asarray(masked)[:args.mask_top])
        assert np.all(np.asarray(masked)[args.mask_top:]==fill)
        checks.append({'event_id':row['event_id'],'image':path.name,'fill_gray':fill,'original_sha256':before})
        if args.mode=='sample':
            panel=Image.new('RGB',(1320,550),'white'); draw=ImageDraw.Draw(panel)
            draw.text((8,5),f"Event {row['event_id']} | {row['image_time_UT']} UT | remaining {row['transit_time_hours']:.6f} h",fill='black')
            for x,text in [(8,'Original 512x512'),(528,'Masked 512x512'),(1050,'CNN input 256x256')]: draw.text((x,22),text,fill='black')
            panel.paste(gray,(0,38)); panel.paste(masked,(520,38)); panel.paste(image,(1048,38))
            comparison=out/'comparisons'/('event_'+str(row['event_id'])+'_'+path.name); comparison.parent.mkdir(exist_ok=True); panel.save(comparison); panels.append(panel)
    write_csv(out/'image_labels.csv',labels,FIELDS); write_csv(out/'excluded_negative_labels.csv',excluded,FIELDS)
    write_csv(out/'excluded_quality_labels.csv',quality_report,FIELDS+['reason','status'])
    info={'mode':args.mode,'ready_for_training':args.mode=='batch' and args.review_approved,'source_images':source_count,'quality_policy_sha256':policy_hash(),'excluded_quality_images':len(quality_report),'quality_negative_overlap':sum(r['status']=='already_excluded_negative' for r in quality_report),'excluded_quality_only_images':sum(r['status']=='excluded_quality' for r in quality_report),'eligible_images':len(rows),'excluded_negative_images':len(excluded),'output_images':len(labels),'mask_box_xyxy_exclusive':[0,args.mask_top,512,512],'resize':'bilinear','normalization':'L / 255 in Dataset; implementation choice','accepted_time_offsets':notes,'checks':checks}
    (out/'preprocessing.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if panels:
        sheet=Image.new('RGB',(1320,550*len(panels)),'white')
        for i,panel in enumerate(panels): sheet.paste(panel,(0,i*550))
        sheet.save(out/'all_comparisons.png')
    print(json.dumps({k:v for k,v in info.items() if k not in ('checks','accepted_time_offsets')},indent=2))
if __name__=='__main__': main()
