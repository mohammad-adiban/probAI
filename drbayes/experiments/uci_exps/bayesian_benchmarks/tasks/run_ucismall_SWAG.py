import subprocess
import os
import sys
import inspect
import stat

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
os.chdir(currentdir)


st = os.stat('run_ucismall_SWAG_naval.sh')
os.chmod('run_ucismall_SWAG_naval.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_concrete.sh')
os.chmod('run_ucismall_SWAG_concrete.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_concrete.sh')
os.chmod('run_ucismall_SWAG_concrete.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_energy.sh')
os.chmod('run_ucismall_SWAG_energy.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_power.sh')
os.chmod('run_ucismall_SWAG_power.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_protein.sh')
os.chmod('run_ucismall_SWAG_protein.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_kin8nm.sh')
os.chmod('run_ucismall_SWAG_kin8nm.sh', st.st_mode | stat.S_IEXEC)


st = os.stat('run_ucismall_SWAG_wine.sh')
os.chmod('run_ucismall_SWAG_wine.sh', st.st_mode | stat.S_IEXEC)



if './naval.txt' == sys.argv[2]:
    print('SELECTED DATASET: naval')
    subprocess.call(currentdir+"/run_ucismall_SWAG_naval.sh", shell=True)
    
    
elif './concrete.txt' == sys.argv[2]:
    print('SELECTED DATASET: concrete')
    subprocess.call(currentdir+"/run_ucismall_SWAG_concrete.sh", shell=True)
    
    
elif './energy.txt' == sys.argv[2]:
    print('SELECTED DATASET: energy')
    subprocess.call(currentdir+"/run_ucismall_SWAG_energy.sh", shell=True)


elif './kin8nm.txt' == sys.argv[2]:
    print('SELECTED DATASET: kin8nm')
    subprocess.call(currentdir+"/run_ucismall_SWAG_kin8nm.sh", shell=True)
    

elif './power.txt' == sys.argv[2]:
    print('SELECTED DATASET: power-plant')
    subprocess.call(currentdir+"/run_ucismall_SWAG_power.sh", shell=True)
    
    
elif './protein.txt' == sys.argv[2]:
    print('SELECTED DATASET: protein-teritary-structure')
    subprocess.call(currentdir+"/run_ucismall_SWAG_protein.sh", shell=True)
    

elif './yacht.txt' == sys.argv[2]:
    print('SELECTED DATASET: yacht')
    subprocess.call(currentdir+"/run_ucismall_SWAG_yacht.sh", shell=True)
    
elif './wine.txt' == sys.argv[2]:
    print('SELECTED DATASET: wine-quality-red')
    subprocess.call(currentdir+"/run_ucismall_SWAG_wine.sh", shell=True)
               
    
elif './concrete.txt' != sys.argv[2] or './energy.txt' != sys.argv[2] or './kin8nm.txt' != sys.argv[2] or './power.txt' != sys.argv[2] or './protein.txt'!= sys.argv[2] or './wine.txt' != sys.argv[2] or './yacht.txt' != sys.argv[2]:
    print('Please make sure you enter the name of the dataset correctly!')
    sys.exit()



