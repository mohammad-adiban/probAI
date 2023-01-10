#python3 main.py --dataset ./power-plant.txt --method 3
#python3 main.py --dataset ./power-plant.txt --method 0
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Improvement on SWAG
if '0' == sys.argv[4]:
        from drbayes.experiments.uci_exps.bayesian_benchmarks.tasks import run_ucismall_improvement

    
    
elif '2' == sys.argv[4]:
    sys.path.insert(0, currentdir+"/mcbn/code/mcbn")
    import main_method2   
        
        
#SWAG:
elif '3' == sys.argv[4]:
    sys.path.insert(0, currentdir+"/drbayes/experiments/uci_exps/bayesian_benchmarks/tasks")
    from drbayes.experiments.uci_exps.bayesian_benchmarks.tasks import run_ucismall_SWAG


else: print('NotImplementedError')

    
   