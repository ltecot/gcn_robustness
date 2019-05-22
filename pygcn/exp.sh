seq 0 10 100 | xargs -I {} -n 1 -P 5 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.01 --dataset pubmed 

#seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.03 --dataset pubmed

#seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file pubmed_16_0_target_nodes.p --epsilon 0.05 --dataset pubmed 

#seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.01 --dataset citeseer --hops 5 

#seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.03 --dataset citeseer --hops 5 
#seq 0 10 100 | xargs -I {} -n 1 -P 11 python test_bound.py --start {} --npoints 10 --p_file citeseer_16_0_target_nodes.p --epsilon 0.05 --dataset citesser --hops 5 
