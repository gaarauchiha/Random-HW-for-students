in_class_exercises = 90
read_review_reflect = 85
problem_sets = [70, 80, 90, 85]
lab_grade = 88
in_class_quizzes = [75, 80, 85, 90]
midterm = 78
final_exam = 92

problem_sets.remove(min(problem_sets))
problem_sets_mean = sum(problem_sets) / len(problem_sets)

in_class_quizzes.remove(min(in_class_quizzes))
in_class_quizzes_mean = sum(in_class_quizzes) / len(in_class_quizzes)

# Final weighted score
final_weighted_score = (in_class_exercises * 0.05 + 
                         read_review_reflect * 0.05 + 
                         problem_sets_mean * 0.05 + 
                         lab_grade * 0.20 + 
                         in_class_quizzes_mean * 0.20 + 
                         midterm * 0.20 + 
                         final_exam * 0.25)

print(final_weighted_score)