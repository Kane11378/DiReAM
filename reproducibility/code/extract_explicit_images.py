"""Safely extract only explicitly named, byte-pinned image/mask members.

No links, arbitrary archive paths, dataset discovery or scientific selection.
RGB and masks are separate invocations. Fitting/prediction never reads masks.
"""
from pathlib import Path
import sys,argparse,tarfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from complete_methods import read,sha,dump
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--kind',choices=['rgb','masks'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=read(a.manifest);source=s['sources'][a.category];assert a.archive.stat().st_size==source['archive_bytes'] and sha(a.archive)==source['archive_sha256'];g=s['groups'][a.category];want={}
 for role in (['reference','calibration','query'] if a.kind=='rgb' else ['query']):
  for row in g[role]:
   if a.kind=='masks' and row['label_stratum']!='anomaly':continue
   name=row['rgb_member' if a.kind=='rgb' else 'gt_member'];expected=row['source_payload_sha256' if a.kind=='rgb' else 'GT_payload_sha256'];assert name not in want or want[name]==expected;want[name]=expected
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);found=set()
 with tarfile.open(a.archive,mode='r|gz') as archive:
  for member in archive:
   if member.name not in want:continue
   assert member.isfile() and not member.issym() and not member.islnk() and 0<member.size<128*1024**2
   target=(root/member.name).resolve();assert target.is_relative_to(root);f=archive.extractfile(member);data=f.read();assert len(data)==member.size and hashlib.sha256(data).hexdigest()==want[member.name]
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():assert sha(target)==want[member.name]
   else:target.write_bytes(data)
   found.add(member.name)
   if len(found)==len(want):break
 assert found==set(want),'Missing explicit archive members';dump(root/(a.category+'_'+a.kind+'_EXTRACTION.json'),dict(status='EXPLICIT_BYTE_IDENTICAL_PAYLOADS_EXTRACTED',archive_sha256=source['archive_sha256'],manifest_sha256=sha(a.manifest),kind=a.kind,members=sorted(found),no_pixel_masks_opened=a.kind=='rgb'))
if __name__=='__main__':main()
