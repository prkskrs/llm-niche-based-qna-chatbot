# Install necessary libraries
# !pip install -U sentence-transformers fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample data
obj = {
      "What courses does your college offer":" Our college offers a wide range of courses. Here is the list of offered courses.\n Architecture\nArtificial Intelligence & Data Science\nBiotechnology\nChemical Engineering\nCivil Engineering\nComputer Science and Engineering\nElectrical and Electronics Engineering\nElectronics and Communication Engineering\nElectronics and Instrumentation Engineering\nElectronics & Telecommunication Engineering\nIndustrial Engineering and Management\nInformation Science and Engineering\nMechanical Engineering\nMaster of Business Administration\nMaster of Computer Application\nNanotechnology\nPhysics\nChemistry\nMathematics",
      "Can you provide information about the faculty members in the Computer Science department?":"Yes, we have a highly qualified faculty team in the Computer Science department. You can find information about each faculty member, including their qualifications and areas of expertise here http://sit.ac.in/html/component/csedept/csedocs/2_Annex_II_2019_20Addedcse.pdf",
      "Can you provide the syllabus for the Computer Science and Engineering course": "The syllabus includes topics covered in each semester, course objectives, and assessment criteria. Please specify which year syllabus do you want?",
      "What is the syllabus of first year of Computer Science and Engineering?": "The syllabus of first year of Computer Science and Engineering is here http://sit.ac.in/html/component/csedept/cseschemasyllabus/I-II%20Sem%20Syllabus%202023-24%20Final.pdf",
      "What is the syllabus of second year of Computer Science and Engineering?":"The syllabus of second year of Computer Science and Engineering is here http://sit.ac.in/html/component/csedept/cseschemasyllabus/CSE%20Syllabus%20-%2022-23%20%20III%20to%20IV%20-%20NEP%202%20print.pdf",
      "What is the syllabus of third year of Computer Science and Engineering?":"The syllabus of third year of Computer Science and Engineering is here http://sit.ac.in/html/component/csedept/cseschemasyllabus/CSE%20160%20credits%20Syllabus%20V%20to%20VI%20NEP%201%20Print.pdf",
      "What is the syllabus of fourth year of Computer Science and Engineering?":"The syllabus of forth year of Computer Science and Engineering is here http://sit.ac.in/html/component/csedept/cseschemasyllabus/CSE%20VII%20VIII%20Batch%202020%20Print.pdf",
      "What is the minimum percentage of attendance required?":"Everyone should maintain 85% and above attendance",
      "Is syllabus of Computer Science and Engineering(CSE) and Artificial Intelligence & Data Science(AI&DS) same?":"No",
      "Is syllabus of Computer Science and Engineering(CSE) and Information Science and Engineering(ISE) same?":"Yes, Syllabus is almost same.",
      "How does IT differ from other comparable colleges?": "The infrastructure at SIT is exceptional, spanning over 50+ acres with numerous facilities such as multiple laboratories for BE students, sports grounds, gyms, and hostels. The college offers extensive training programs including personality development, soft skills, aptitude tests, and technical workshops. The curriculum is rigorous with strict attendance requirements and continuous evaluations through quizzes, internals, assignments, and exams. The faculty in the Computer Science department is highly qualified, and the college provides ample opportunities for research and internships. Placement opportunities are abundant, with companies visiting as early as the 7th semester and offering attractive salary packages. Overall, SIT is renowned for its infrastructure, training programs, curriculum, faculty, research opportunities, and placements, making it one of the top colleges in Karnataka.",
      "What sets this college apart from other colleges?":"The infrastructure at SIT is exceptional, spanning over 50+ acres with numerous facilities such as multiple laboratories for BE students, sports grounds, gyms, and hostels. The college offers extensive training programs including personality development, soft skills, aptitude tests, and technical workshops. The curriculum is rigorous with strict attendance requirements and continuous evaluations through quizzes, internals, assignments, and exams. The faculty in the Computer Science department is highly qualified, and the college provides ample opportunities for research and internships. Placement opportunities are abundant, with companies visiting as early as the 7th semester and offering attractive salary packages. Overall, SIT is renowned for its infrastructure, training programs, curriculum, faculty, research opportunities, and placements, making it one of the top colleges in Karnataka.",
      "What extracurricular activities or student groups are available?":"At SIT, there is a vibrant array of extracurricular activities and student groups that cater to various interests. For theater enthusiasts, there are groups like Black Pearl, Aurora, and Drushya that showcase talent and creativity on stage. Technical enthusiasts can join Decoders, Sark, and Scriptink to delve into coding, programming, and technical projects. Robotics enthusiasts have a platform in Corsit, while dance enthusiasts can join clubs like Panthers and Vulcans. These clubs have a rich history of winning prizes and accolades, bringing pride to the college through their achievements.",
      "What financial aid or scholarship opportunities are available?":"Yes, If you are good at academics you will get financial aid and scholarship opportunities. You will be informed once oppotunities are available.",
      "What are the courses being offered?": " Our college offers a wide range of courses. Here is the list of offered courses.\n Architecture\nArtificial Intelligence & Data Science\nBiotechnology\nChemical Engineering\nCivil Engineering\nComputer Science and Engineering\nElectrical and Electronics Engineering\nElectronics and Communication Engineering\nElectronics and Instrumentation Engineering\nElectronics & Telecommunication Engineering\nIndustrial Engineering and Management\nInformation Science and Engineering\nMechanical Engineering\nMaster of Business Administration\nMaster of Computer Application\nNanotechnology\nPhysics\nChemistry\nMathematics",
      "Can we pay online the admission fees ?": "Yes, Any fees can be paid online by visiting http://erp.sit.ac.in .",
      "Which courses are high in demand at SIT?":"Computer Science and Engineering are high in demand.",
      "What is the college's ranking in engineering education?": "SIT Tumkur has been granted autonomous status by UGC and Government of Karnataka. As per NIRF rankings 2023, the Institute falls in the 101-150 rank band. The Institute is accredited by NAAC with A++ Grade.",
      "Are there opportunities for hands-on projects and practical learning in engineering?": "Yes, at our college, we offer extensive opportunities for hands-on projects and practical learning in engineering. We organize numerous hackathons and technical competitions where students can showcase their skills, collaborate with peers, and solve real-world problems. Additionally, our engineering syllabus includes both mini and major projects that are designed to provide practical learning experiences. These projects allow students to apply theoretical knowledge in a hands-on environment, develop problem-solving skills, and gain valuable industry-relevant experience. Overall, our emphasis on hackathons, technical competitions, and project-based learning ensures that students receive comprehensive practical exposure during their engineering education.",
      "How does the college support engineering students in internships and industry placements?": "The college includes soft skills training in the syllabus, enhancing students' professional capabilities for internships and industry placements.In addition to soft skills training in the syllabus, our college also offers industrial training programs. These programs provide students with hands-on experience in industrial settings, allowing them to apply theoretical knowledge to practical scenarios and gain valuable insights into industry practices. This combination of soft skills training and industrial training prepares our engineering students thoroughly for internships and industry placements, equipping them with the necessary skills and confidence to excel in their professional endeavors.",
      "What research opportunities are available for engineering students?":"Engineering students at our college have ample research opportunities, including the chance to work alongside professors on innovative projects. These opportunities allow students to explore their interests, contribute to cutting-edge research, and gain valuable experience in their field of study. By collaborating with professors, students can delve into advanced topics, publish papers, and even participate in conferences, enhancing their academic and professional profiles. This hands-on approach to research fosters a culture of innovation and inquiry, preparing students for future careers in academia or industry.",
      "Are there engineering clubs or student organizations on campus?": "At SIT, there is a vibrant array of extracurricular activities and student groups that cater to various interests. For theater enthusiasts, there are groups like Black Pearl, Aurora, and Drushya that showcase talent and creativity on stage. Technical enthusiasts can join Decoders, Sark, and Scriptink to delve into coding, programming, and technical projects. Robotics enthusiasts have a platform in Corsit, while dance enthusiasts can join clubs like Panthers and Vulcans. These clubs have a rich history of winning prizes and accolades, bringing pride to the college through their achievements.",
      "Can you describe the campus environment for engineering students?": "The campus environment for engineering students is dynamic and conducive to academic excellence. It offers state-of-the-art facilities, including well-equipped laboratories, modern classrooms, and dedicated study spaces. The campus fosters a culture of innovation and collaboration, encouraging students to engage in hands-on projects, research activities, and technical competitions. Additionally, the presence of industry partnerships, guest lectures, and workshops enriches the learning experience for engineering students, providing practical insights into real-world challenges. The campus also promotes a healthy work-life balance with recreational facilities, sports amenities, and vibrant student clubs, creating a holistic environment for personal and professional growth.",
      "Can engineering students participate in industry-sponsored projects or research?": "Yes, engineering students can actively participate in industry-sponsored projects or research initiatives. These projects are often collaborative efforts between the college and industry partners, providing students with valuable opportunities to work on real-world problems, gain hands-on experience, and apply their academic knowledge in practical settings. By engaging in industry-sponsored projects or research, students can develop industry-relevant skills, build professional networks, and enhance their career prospects. These experiences also contribute to a well-rounded education, preparing students for success in their chosen fields upon graduation.",
      "Are there opportunities for engineering students to present their work at conferences or symposiums?": "Yes, there are ample opportunities for engineering students to present their work at conferences or symposiums. Our college actively encourages students to showcase their research, projects, and innovations at various academic events both within the college and at external conferences. These opportunities not only allow students to share their work with a wider audience but also provide valuable networking and learning experiences. Presenting at conferences or symposiums enables students to receive feedback, gain recognition for their achievements, and stay updated with the latest developments in their fields of interest. ",
      "Who created you?": "I am created by Smita Sen, Sneha Singh, Tushant Arora and Vinit Kumar.",
      "Who is your developer?": "I am created by Smita Sen, Sneha Singh, Tushant Arora and Vinit Kumar.",
      "Who made you?": "I am created by Smita Sen, Sneha Singh, Tushant Arora and Vinit Kumar.",
      "What is your name?": "You can call me Buddy. I'm a Chatbot.",
      "What should I call you?": "You can call me Buddy. I'm a Chatbot.",
      "What is your name?":"You can call me Buddy I'm a Chatbot.",
      "What are the college timings?": "The college is open from 8am to 5pm, Monday to Saturday.",
      "When is the college open?": "The college is open from 8am to 5pm, Monday to Saturday.",
      "What is the college telephone number?": "You can contact the college at +918162282696, +918162214001, +918162282994, +919071312027 or you can mail at principal@sit.ac.in",
      "Can I get your contact number?": "You can contact the college at +918162282696, +918162214001, +918162282994, +919071312027 or you can mail at principal@sit.ac.in",
      "What are the branches in the college":" Our college offers a wide range of courses. Here is the list of offered courses.\n Architecture\nArtificial Intelligence & Data Science\nBiotechnology\nChemical Engineering\nCivil Engineering\nComputer Science and Engineering\nElectrical and Electronics Engineering\nElectronics and Communication Engineering\nElectronics and Instrumentation Engineering\nElectronics & Telecommunication Engineering\nIndustrial Engineering and Management\nInformation Science and Engineering\nMechanical Engineering\nMaster of Business Administration\nMaster of Computer Application\nNanotechnology\nPhysics\nChemistry\nMathematics",
      "What are the branches in the college": " Our college offers a wide range of courses. Here is the list of offered courses.\n Architecture\nArtificial Intelligence & Data Science\nBiotechnology\nChemical Engineering\nCivil Engineering\nComputer Science and Engineering\nElectrical and Electronics Engineering\nElectronics and Communication Engineering\nElectronics and Instrumentation Engineering\nElectronics & Telecommunication Engineering\nIndustrial Engineering and Management\nInformation Science and Engineering\nMechanical Engineering\nMaster of Business Administration\nMaster of Computer Application\nNanotechnology\nPhysics\nChemistry\nMathematics",
      "What are the hostel fees?": "For detailed fee information, please visit our college website.",
      "Tell me about the fees.": "For detailed fee information, please visit our college website.",
      "How much is the college fee?": "For detailed fee information, please visit our college website.",
      "Where is the library located?": "Yes, the college has a library. The library is located on right side of Admin Block. The timings are from 9am to 5pm, Monday to Friday.",
      "Does the college have a library?": "Yes, the college has a library. The library is located on right side of Admin Block. The timings are from 9am to 5pm, Monday to Friday.",
      "What are the library timings?": "Yes, the college has a library. The library is located on right side of Admin Block. The timings are from 9am to 5pm, Monday to Friday.",
      "Are you happy here?": "As an AI, I don't have emotions, but I'm here to assist and provide information about the college.",
      "Do you enjoy being at this college?":"As an AI, I don't have emotions, but I'm here to assist and provide information about the college.",
      "How accessible are administrators, registrars, financial aid officers, etc.? ":"The college administration strives to maintain accessibility for students. Office hours and contact information for various staff members are available for student assistance. ",
      " Can students easily reach out to college staff when needed?":" The college administration strives to maintain accessibility for students. Office hours and contact information for various staff members are available for student assistance.",
      "What majors are popular? ":"Popular majors in our college include Computer Science, Information Science and Artificial Intelligence.",
      "Which fields of study have a high enrollment?":"Popular majors in our college include Computer Science, Information Science and Artificial Intelligence. ",
      "What are the most sought-after majors in your college? ":" Popular majors in our college include Computer Science, Information Science and Artificial Intelligence.",
      "Are your classes lecture-based or discussion-based? ":"The class format can vary depending on the subject and the instructor's teaching style. Some classes may be lecture-based, while others may involve more interactive discussions and group activities. ",
      " How are the classes structured?":" The class format can vary depending on the subject and the instructor's teaching style. Some classes may be lecture-based, while others may involve more interactive discussions and group activities.",
      "What's the typical format of a class? ":"The class format can vary depending on the subject and the instructor's teaching style. Some classes may be lecture-based, while others may involve more interactive discussions and group activities. ",
      "What is the syllabus of first year of Information Science and Engineering": "To check our updated syllabus, Please visit out college website.",
      "What is the syllabus of fourth year of Information Science and Engineering": "To check our updated syllabus, Please visit out college website.",
      "What is the syllabus of second year of Information Science and Engineering": "To check our updated syllabus, Please visit out college website.",
      "What is the syllabus of third year of Information Science and Engineering": "To check our updated syllabus, Please visit out college website.",
      "Are professors available for research with students? ":"Faculty members at our college actively engage in research, and students often have opportunities to collaborate on research projects. You can reach out to professors in your field of interest to inquire about research opportunities. ",
      "How can I get involved in research at your college?":"Faculty members at our college actively engage in research, and students often have opportunities to collaborate on research projects. You can reach out to professors in your field of interest to inquire about research opportunities. ",
      "Can students engage in research projects with faculty?":"Faculty members at our college actively engage in research, and students often have opportunities to collaborate on research projects. You can reach out to professors in your field of interest to inquire about research opportunities. ",
      "What's it like to be a first-year student here? ":"As a first-year student, you can expect a supportive and welcoming environment. Orientation programs, peer mentoring, and academic support services are available to help you transition smoothly into college life. ",
      "Can you describe the first-year experience?":"As a first-year student, you can expect a supportive and welcoming environment. Orientation programs, peer mentoring, and academic support services are available to help you transition smoothly into college life. ",
      "What should I expect as a freshman?":"As a first-year student, you can expect a supportive and welcoming environment. Orientation programs, peer mentoring, and academic support services are available to help you transition smoothly into college life. ",
      "What's a typical day like? ":"A typical day as a student involves attending classes, engaging in study sessions, participating in extracurricular activities, and utilizing campus resources. Each student's daily routine may vary based on their schedule and interests. ",
      "Can you describe a day in the life of a student? ":"A typical day as a student involves attending classes, engaging in study sessions, participating in extracurricular activities, and utilizing campus resources. Each student's daily routine may vary based on their schedule and interests. ",
      "What's the daily routine at your college? ":"A typical day as a student involves attending classes, engaging in study sessions, participating in extracurricular activities, and utilizing campus resources. Each student's daily routine may vary based on their schedule and interests. ",
      "What do you do when you're not in class? ":"Outside of class, students engage in various activities such as participating in clubs and organizations, attending campus events, exploring the surrounding community, and spending time with friends. ",
      "How do students spend their free time? ":"Outside of class, students engage in various activities such as participating in clubs and organizations, attending campus events, exploring the surrounding community, and spending time with friends. ",
      "Are there activities to do outside of class? ":"Outside of class, students engage in various activities such as participating in clubs and organizations, attending campus events, exploring the surrounding community, and spending time with friends. ",
      "Where do you like to study? ":"There are several study spots on campus, including libraries, coffee shops, and outdoor seating areas. It's important to find a location that suits your study preferences and helps you focus. ",
      "What are some good study spots? ":"There are several study spots on campus, including libraries, coffee shops, and outdoor seating areas. It's important to find a location that suits your study preferences and helps you focus. ",
      "Are there quiet places to study on campus? ":"There are several study spots on campus, including libraries, coffee shops, and outdoor seating areas. It's important to find a location that suits your study preferences and helps you focus. ",
      "why should be take admission in your college?":"The infrastructure at SIT is exceptional, spanning over 50+ acres with numerous facilities such as multiple laboratories for BE students, sports grounds, gyms, and hostels. The college offers extensive training programs including personality development, soft skills, aptitude tests, and technical workshops. The curriculum is rigorous with strict attendance requirements and continuous evaluations through quizzes, internals, assignments, and exams. The faculty in the Computer Science department is highly qualified, and the college provides ample opportunities for research and internships. Placement opportunities are abundant, with companies visiting as early as the 7th semester and offering attractive salary packages. Overall, SIT is renowned for its infrastructure, training programs, curriculum, faculty, research opportunities, and placements, making it one of the top colleges in Karnataka. "
     
    }


# Encode questions
tempLst = [model.encode(i) for i in obj]

# Function to compute similarity
def getSimilarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Function to get similar question
def getSimilarQuestion(q1):
    score = [getSimilarity(i, model.encode(q1)) for i in tempLst]
    return obj[list(obj.keys())[score.index(max(score))]]

# Define request body model
class Question(BaseModel):
    question: str

# Define endpoint
@app.post("/get_answer/")
def get_answer(question: Question):
    return {"answer": getSimilarQuestion(question.question)}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

