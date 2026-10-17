"""The bounded bridge must preserve native rg behavior and argv on Windows."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evals'))
from native_rg import native_rg


@pytest.mark.skipif(os.name!='nt',reason='Windows native argv bridge')
@pytest.mark.parametrize('pattern',['^alpha|^beta','a"b',r'path\\',r'^中文 带 空格$','literal [&<>]','', 'not-found','['])
def test_native_bridge_matches_real_rg_with_shell_characters_and_unicode(tmp_path,pattern):
    root=Path(__file__).resolve().parents[1];binary=tmp_path/'bin';binary.mkdir()
    real=shutil.which('rg');assert real
    shutil.copyfile(native_rg(root),binary/'rg.exe')
    shutil.copyfile(real,binary/'skillforge-rg.exe')
    for name in ('bounded_command.py','public_output.py'):shutil.copyfile(root/'evals'/name,binary/name)
    target=tmp_path/'中文 文件.txt'
    target.write_text('alpha\nbeta\na"b\npath\\\n中文 带 空格\nliteral &<>\n','utf-8')
    argv=['-n','-e',pattern,'--',str(target)]
    env=dict(os.environ,VIRTUAL_ENV=str(Path(sys.executable).parent.parent),
             PYTHONIOENCODING='utf-8',TEMP=str(tmp_path))
    expected=subprocess.run([real,*argv],capture_output=True,env=env,timeout=10)
    actual=subprocess.run([str(binary/'rg.exe'),*argv],capture_output=True,env=env,timeout=10)
    assert actual.returncode==expected.returncode
    assert actual.stdout==expected.stdout and actual.stderr==expected.stderr
