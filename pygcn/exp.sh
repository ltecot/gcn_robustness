seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.01 --dataset cora 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.03 --dataset cora 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.05 --dataset cora 

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.01 --dataset cora --single 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.03 --dataset cora --single
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.05 --dataset cora --single

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.01 --dataset cora --hops 2 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.03 --dataset cora --hops 2
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.05 --dataset cora --hops 2

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.01 --dataset cora --hops 5 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.03 --dataset cora --hops 5
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file cora_16_0_target_nodes.p --epsilon 0.05 --dataset cora --hops 5


seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.001 --dataset pubmed --single 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.005 --dataset pubmed --single 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed --single 

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.001 --dataset pubmed --hops 2 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.005 --dataset pubmed --hops 2 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed --hops 2 

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed 

seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.001 --dataset pubmed --hops 5 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.005 --dataset pubmed --hops 5 
seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed --hops 5 


seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.01 --dataset citeseer --hops 5 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.02 --dataset citeseer --hops 5 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.03 --dataset citesser --hops 5 

seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.01 --dataset citeseer --hops 2 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.02 --dataset citeseer --hops 2 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.03 --dataset citesser --hops 2 

seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.01 --dataset citeseer --hops 1 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.02 --dataset citeseer --hops 1 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.03 --dataset citesser --hops 1 

seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.01 --dataset citeseer --single 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.02 --dataset citeseer --single 
seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.03 --dataset citesser --single 
