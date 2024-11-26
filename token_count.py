import tiktoken

text = """
question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
 inference: Number of clips sold in April: 48
Number of clips sold in May: 48 / 2 = 24
Total number of clips sold: 48 + 24 = 72

Final Answer: The final answer is $72$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Number of clips sold in April: 48
Number of clips sold in May: 48 / 2 = 24
Total number of clips sold: 48 + 24 = 72

Final Answer: The final answer is $72$. I hope it is correct.

 inference: Final Answer: The final answer is $72$. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
 inference: Weng earns $12 per hour.
There are 60 minutes in an hour.
She babysat for 50 minutes.
Her hourly rate is $12/60 minutes = $0.20 per minute.
She earned $0.20/minute * 50 minutes = $10.

Final Answer: The final answer is $10. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Weng earns $12 per hour.
There are 60 minutes in an hour.
She babysat for 50 minutes.
Her hourly rate is $12/60 minutes = $0.20 per minute.
She earned $0.20/minute * 50 minutes = $10.

Final Answer: The final answer is $10. I hope it is correct.

 inference: Final Answer: The final answer is $10. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
 inference: Betty needs $100 for the wallet.
She has half the money, so she has $100/2 = $50.
Her parents gave her $15.
Her grandparents gave her 2 * $15 = $30.
Total money Betty has: $50 + $15 + $30 = $95.
Money Betty still needs: $100 - $95 = $5.

Final Answer: The final answer is $5. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Betty needs $100 for the wallet.
She has half the money, so she has $100/2 = $50.
Her parents gave her $15.
Her grandparents gave her 2 * $15 = $30.
Total money Betty has: $50 + $15 + $30 = $95.
Money Betty still needs: $100 - $95 = $5.

Final Answer: The final answer is $5. I hope it is correct.

 inference: Final Answer: The final answer is $5. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? 
 inference: Pages read yesterday: 12 pages
Pages read today: 12 pages * 2 = 24 pages
Total pages read: 12 pages + 24 pages = 36 pages
Remaining pages: 120 pages - 36 pages = 84 pages
Pages to read tomorrow: 84 pages / 2 = 42 pages

Final Answer: The final answer is $42$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Pages read yesterday: 12 pages
Pages read today: 12 pages * 2 = 24 pages
Total pages read: 12 pages + 24 pages = 36 pages
Remaining pages: 120 pages - 36 pages = 84 pages
Pages to read tomorrow: 84 pages / 2 = 42 pages

Final Answer: The final answer is $42$. I hope it is correct.

 inference: Final Answer: The final answer is $42$. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?       
 inference: Pages per letter: 3
Letters per week: 2
Pages per week: 3 pages/letter * 2 letters/week = 6 pages/week
Weeks per year: 52
Total pages per year: 6 pages/week * 52 weeks/year = 312 pages/year

Final Answer: The final answer is $312$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Pages per letter: 3
Letters per week: 2
Pages per week: 3 pages/letter * 2 letters/week = 6 pages/week
Weeks per year: 52
Total pages per year: 6 pages/week * 52 weeks/year = 312 pages/year

Final Answer: The final answer is $312$. I hope it is correct.

 inference: Final Answer: The final answer is $312$. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
 inference: Purple flowers: 10 * 1.8 = 18
Total yellow and purple flowers: 10 + 18 = 28
Green flowers: 28 * 0.25 = 7
Total flowers: 10 + 18 + 7 = 35

Final Answer: The final answer is $35$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Purple flowers: 10 * 1.8 = 18
Total yellow and purple flowers: 10 + 18 = 28
Green flowers: 28 * 0.25 = 7
Total flowers: 10 + 18 + 7 = 35

Final Answer: The final answer is $35$. I hope it is correct.

 inference: Final Answer: The final answer is $35$. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?   
 inference: Large pizzas: 2 * 16 slices = 32 slices
Small pizzas: 2 * 8 slices = 16 slices
Total slices: 32 + 16 = 48 slices

Final Answer: The final answer is $48$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Large pizzas: 2 * 16 slices = 32 slices
Small pizzas: 2 * 8 slices = 16 slices
Total slices: 32 + 16 = 48 slices

Final Answer: The final answer is $48$. I hope it is correct.

 inference: Final Answer: The final answer is $48$. I hope it is correct.

question: You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?
 inference: Initially, the weight is 2 pounds.

After adding brownies, the weight triples: 2 * 3 = 6 pounds.

After adding 2 pounds of jelly beans, the weight becomes: 6 + 2 = 8 pounds.

After adding gummy worms, the weight doubles: 8 * 2 = 16 pounds.

Final Answer: The final answer is $16$. I hope it is correct.

question: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
Initially, the weight is 2 pounds.

After adding brownies, the weight triples: 2 * 3 = 6 pounds.

After adding 2 pounds of jelly beans, the weight becomes: 6 + 2 = 8 pounds.

After adding gummy worms, the weight doubles: 8 * 2 = 16 pounds.

Final Answer: The final answer is $16$. I hope it is correct.

 inference: Final Answer: The final answer is $16$. I hope it is correct.
"""

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Count the number of tokens
tokens = tokenizer.encode(text)
num_tokens = len(tokens)

print(f"Number of tokens: {num_tokens}")