s = tf('s');  
delay_approx = (1 - 2*s) / (1 + 2*s); 
K = 0.0858;
G = K * delay_approx / s;
num = [ -2 1 ]; den = [2 1 0];
rlocus(num, den); 
