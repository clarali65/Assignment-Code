TEC_test.m and TEC_test_rot.m are the main functions.

TEC_test.m is for the unrotated problem and TEC_test_rot.m is for the rotated ones.

TEC_test_function.m includes the benchmark functions, and the data files are the corrsponding data for the benchmark functions.


If you want to try other algorithms, you can call the functions in the same way in TEC_test.m and TEC_test_rot.m

Group A(use TEC_test.m,funchoose=[1,2]):
1. sphere 
2. rosenbrock

Group B(use TEC_test.m,funchoose=[3,4,8,5,6,7]):
3. ackley
4. griewank
8. weierstrass
5. rastrigin
6. rastrigin_noncont
7. schewfel

Group C(use TEC_test_rot.m,funchoose=[3,4,8,5,6,7]):
3. ackley
4. griewank
8. weierstrass
5. rastrigin
6. rastrigin_noncont
7. schewfel

Group D:(use TEC_test.m,funchoose=[13,15]):
13. composition function 1. com_func1(x);   
15. composition function 2. hybrid_func2(x);

