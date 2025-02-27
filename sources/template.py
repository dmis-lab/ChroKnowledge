from liquid import Template

### System ###
general_system = '''You are a helpful knowledge expert in various domains, and your task is to answer an object following subject, and relation. Please first think step-by-step and generate the answer in the position [Object]. Organize your output in an answer format, "A. [Object]". Your responses will be used for research purposes only, so please have a definite and detail answer.'''
chrono_none_system = """Answer 'Candidate A. [Object]' based on the timestamp. Output only the answer, 'A. [Object]'."""
chrono_cand_system = """Answer 'Candidate A. [Object]' based on the timestamp. If it is correct, repeat same [Object]. If it is wrong, generate new [Object]. Output only the answer, 'A. [Object]'."""

### Prompt of Generation (Knowledge Triplet) ###

prompt_KT_zs = Template("""Q. {{sub}}, {{rel}}, [Object]
""")

prompt_KT_zs_t = Template("""Q. In {{t}}, {{sub}}, {{rel}}, [Object]
""")

prompt_KT = Template("""Q. {{sub1}}, {{rel1}}, [Object]
A. {{obj1}}

Q. {{sub2}}, {{rel2}}, [Object]
A. {{obj2}}

Q. {{sub3}}, {{rel3}}, [Object]
A. {{obj3}}

Q. {{sub4}}, {{rel4}}, [Object]
A. {{obj4}}

Q. {{sub}}, {{rel}}, [Object]
""")

prompt_KT_t = Template("""Q. In {{t1}}, {{sub1}}, {{rel1}}, [Object]
A. {{obj1}}

Q. In {{t2}}, {{sub2}}, {{rel2}}, [Object]
A. {{obj2}}

Q. In {{t3}}, {{sub3}}, {{rel3}}, [Object]
A. {{obj3}}

Q. In {{t4}}, {{sub4}}, {{rel4}}, [Object]
A. {{obj4}}

Q. In {{t}}, {{sub}}, {{rel}}, [Object]
""")

prompt_GEN_zs = Template("""Q. {{q}}
""")

prompt_GEN_zs_t = Template("""Q. In {{t}}, {{q}}
""")

prompt_GEN_Legal = Template("""Q. {{q1}}
A. {{a1}}

Q. {{q2}}
A. {{a2}}

Q. {{q3}}
A. {{a3}}

Q. {{q4}}
A. {{a4}}

Q. {{q}}
""")

prompt_GEN_Legal_t = Template("""Q. In {{t1}}, {{q1}}
A. {{a1}}

Q. In {{t2}}, {{q2}}
A. {{a2}}

Q. In {{t3}}, {{q3}}
A. {{a3}}

Q. In {{t4}}, {{q4}}
A. {{a4}}

Q. In {{t}}, {{q}}
""")

### Prompt of TF (True False) ###

# prompt_TF_Biomedical_t = {
#     "isa": Template("""Q. In {{t}}, {{sub}} is a {{obj}}.
# """),
#     "inverse isa": Template("""Q. In {{t}}, {{sub}} is an inverse of {{obj}}.
# """),
#     "direct procedure site of": Template("""Q. In {{t}}, {{sub}} is direct procedure site of {{obj}}.
# """),
#     "indirect procedure site of": Template("""Q. In {{t}}, {{sub}} is indirect procedure site of {{obj}}.
# """),
#     "is primary anatomic site of disease": Template("""Q. In {{t}}, {{sub}} is primary anatomic site of disease {{obj}}.
# """),
#     "is not primary anatomic site of disease": Template("""Q. In {{t}}, {{sub}} is not the primary anatomic site of disease {{obj}}.
# """),
#     "is normal tissue origin of disease": Template("""Q. In {{t}}, {{sub}} is normal tissue origin of disease {{obj}}.
# """),
#     "is not normal tissue origin of disease": Template("""Q. In {{t}}, {{sub}} is not normal tissue origin of disease {{obj}}.
# """),
#     "is normal cell origin of disease": Template("""Q. In {{t}}, {{sub}} is normal cell origin of disease {{obj}}.
# """),
#     "is not normal cell origin of disease": Template("""Q. In {{t}}, {{sub}} is not normal cell origin of disease {{obj}}.
# """),
#     "has direct procedure site": Template("""Q. In {{t}}, {{sub}} has direct procedure site {{obj}}.
# """),
#     "has indirect procedure site": Template("""Q. In {{t}}, {{sub}} has indirect procedure site {{obj}}.
# """),
#     "positively regulates": Template("""Q. In {{t}}, {{sub}} positively regulates {{obj}}.
# """),
#     "negatively regulates": Template("""Q. In {{t}}, {{sub}} negatively regulates {{obj}}.
# """)
# }

prompt_TF_Biomedical_t = {
    "isa": Template("""Q. In {{t}}, {{sub}} is a {{obj}}.
"""),
    "inverse_isa": Template("""Q. In {{t}}, {{sub}} is an inverse of {{obj}}.
"""),
    "direct_procedure_site_of": Template("""Q. In {{t}}, {{sub}} is direct procedure site of {{obj}}.
"""),
    "has_direct_procedure_site": Template("""Q. In {{t}}, {{sub}} has direct procedure site {{obj}}.
"""),
    "may_prevent": Template("""Q. In {{t}}, {{sub}} may prevent {{obj}}.
"""),
    "may_be_prevented_by": Template("""Q. In {{t}}, {{sub}} may be prevented by {{obj}}.
"""),
    "may_treat": Template("""Q. In {{t}}, {{sub}} may treat {{obj}}.
"""),
    "may_be_treated_by": Template("""Q. In {{t}}, {{sub}} may be treated by {{obj}}.
"""),
    "induces": Template("""Q. In {{t}}, {{sub}} induces {{obj}}.
"""),
    "causative_agent_of": Template("""Q. In {{t}}, {{sub}} is causative agent of {{obj}}.
"""),
    "has_mechanism_of_action": Template("""Q. In {{t}}, {{sub}} has mechanism of action {{obj}}.
"""),
    "has_physiologic_effect": Template("""Q. In {{t}}, {{sub}} has physiologic effect {{obj}}.
""")
}

prompt_TF_General_t = {
    "employer": Template("""Q. In {{t}}, {{sub}} is an employer of {{obj}}.
"""),
    "member of sports team": Template("""Q. In {{t}}, {{sub}} is a member of the sports team {{obj}}.
"""),
    "officeholder": Template("""Q. In {{t}}, {{sub}} is an officeholder of {{obj}}.
"""),
    "member of": Template("""Q. In {{t}}, {{sub}} is a member of {{obj}}.
"""),
    "educated at": Template("""Q. In {{t}}, {{sub}} was educated at {{obj}}.
"""),
    "chief executive officer": Template("""Q. In {{t}}, {{sub}} is the chief executive officer of {{obj}}.
"""),
    "director / manager": Template("""Q. In {{t}}, {{sub}} is the director or manager of {{obj}}.
"""),
    "position held": Template("""Q. In {{t}}, {{sub}} holds the position of {{obj}}.
""")
}

prompt_TF_Legal_t = Template("""Please answer true/false.
Q. In {{t}}, {{q}}
""")

prompt_TF_CommonSense_t = {
    "HasProperty": Template("""Q. In {{t}}, {{sub}} has property {{obj}}.
"""),
    "NotHasProperty": Template("""Q. In {{t}}, {{sub}} not has property {{obj}}.
"""),
    "CapableOf": Template("""Q. In {{t}}, {{sub}} is capable of {{obj}}.
"""),
    "NotCapableOf": Template("""Q. In {{t}}, {{sub}} is not capable of {{obj}}.
"""),
    "Desires": Template("""Q. In {{t}}, {{sub}} desires {{obj}}.
"""),
    "NotDesires": Template("""Q. In {{t}}, {{sub}} does not desires {{obj}}.
"""),
    "Synonym": Template("""Q. In {{t}}, {{sub}} is synonym for {{obj}}.
"""),
    "Antonym": Template("""Q. In {{t}}, {{sub}} is antonym for {{obj}}.
""")
}

prompt_TF_Math_t = {
    "Synonym": Template("""Q. In {{t}}, {{sub}} is synonym for {{obj}}.
"""),
    "Antonym": Template("""Q. In {{t}}, {{sub}} is antonym for {{obj}}.
"""),
    "Rely_on": Template("""Q. In {{t}}, {{sub}} rely on {{obj}}.
"""),
    "Inverse_rely_on": Template("""Q. In {{t}}, {{sub}} is relied on by {{obj}}.
"""),
    "Belong_to": Template("""Q. In {{t}}, {{sub}} belong to {{obj}}.
"""),
    "Inverse_belong_to": Template("""Q. In {{t}}, {{sub}} is belonged to by {{obj}}.
"""),
    "Property_of": Template("""Q. In {{t}}, {{sub}} is a property of {{obj}}.
"""),
    "Inverse_property_of": Template("""Q. In {{t}}, {{sub}} has the property of {{obj}}.
"""),
    "None": Template("""Q. In {{t}}, {{sub}} has no specific relation with {{obj}}.
"""),
    "Similar": Template("""Q. In {{t}}, {{sub}} is similar to {{obj}}.
"""),
    "Apposition": Template("""Q. In {{t}}, {{sub}} is in apposition to {{obj}}.
"""),
    "Other": Template("""Q. In {{t}}, {{sub}} has other relations with {{obj}}.
"""),
}


### Prompt of MCQA (Question Answer) ###

# prompt_QA_Biomedical = {
#     "isa": Template("""Which of the following is classified as a {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "inverse isa": Template("""Which of the following is classified as an inverse of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "direct procedure site of": Template("""What is direct procedure site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "indirect procedure site of": Template("""What is indirect procedure site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is primary anatomic site of disease": Template("""What is the primary anatomic site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is not primary anatomic site of disease": Template("""What is not the primary anatomic site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is normal tissue origin of disease": Template("""What is the normal tissue of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is not normal tissue origin of disease": Template("""What is not the normal tissue of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is normal cell origin of disease": Template("""What is the normal cell of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is not normal cell origin of disease": Template("""What is not the normal cell of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "has direct procedure site": Template("""What has direct procedure site {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "has indirect procedure site": Template("""What has indirect procedure site {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "positively regulates": Template("""What positively regulates {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "negatively regulates": Template("""What negatively regulates {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
# }

prompt_QA_Biomedical = {
   "isa": Template("""Which of the following is classified as a {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "inverse_isa": Template("""Which of the following is classified as an inverse of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "direct_procedure_site_of": Template("""What is direct procedure site of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "has_direct_procedure_site": Template("""What has direct procedure site {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_prevent": Template("""Which of the following may prevent {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_be_prevented_by": Template("""Which of the following does {{sub}} may prevent?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "may_treat": Template("""Which of the following may treat {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_be_treated_by": Template("""Which of the following does {{sub}} may treat?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "induces": Template("""Which of the following induces {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "causative_agent_of": Template("""What is {{sub}} the causative agent of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "has_mechanism_of_action": Template("""What is the mechanism of action of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "has_physiologic_effect": Template("""What is the physiologic effect of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}

# prompt_QA_Biomedical_t = {
#     "isa": Template("""In {{t}}, which of the following is classified as a {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "inverse isa": Template("""In {{t}}, which of the following is classified as an inverse of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "direct procedure site of": Template("""In {{t}}, what is direct procedure site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "indirect procedure site of": Template("""In {{t}}, what is indirect procedure site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is primary anatomic site of disease": Template("""In {{t}}, what is the primary anatomic site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is not primary anatomic site of disease": Template("""In {{t}}, what is not the primary anatomic site of {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "is normal tissue origin of disease": Template("""In {{t}}, what is the normal tissue of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is not normal tissue origin of disease": Template("""In {{t}}, what is not the normal tissue of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is normal cell origin of disease": Template("""In {{t}}, what is the normal cell of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "is not normal cell origin of disease": Template("""In {{t}}, what is not the normal cell of origin for {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),# reversed
#     "has direct procedure site": Template("""In {{t}}, what has direct procedure site {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "has indirect procedure site": Template("""In {{t}}, what has indirect procedure site {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "positively regulates": Template("""In {{t}}, what positively regulates {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
#     "negatively regulates": Template("""In {{t}}, what negatively regulates {{sub}}?
# (a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
# """),
# }

prompt_QA_Biomedical_t = {
   "isa": Template("""In {{t}}, which of the following is classified as a {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "inverse_isa": Template("""In {{t}}, which of the following is classified as an inverse of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "direct_procedure_site_of": Template("""In {{t}}, what is direct procedure site of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "has_direct_procedure_site": Template("""In {{t}}, what has direct procedure site {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_prevent": Template("""In {{t}}, which of the following may prevent {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_be_prevented_by": Template("""In {{t}}, which of the following does {{sub}} may prevent?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "may_treat": Template("""In {{t}}, which of the following may treat {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "may_be_treated_by": Template("""In {{t}}, which of the following does {{sub}} may treat?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "induces": Template("""In {{t}}, which of the following induces {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "causative_agent_of": Template("""In {{t}}, what is {{sub}} the causative agent of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""), # reversed
   "has_mechanism_of_action": Template("""In {{t}}, what is the mechanism of action of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
   "has_physiologic_effect": Template("""In {{t}}, what is the physiologic effect of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}

prompt_QA_General = {
    "employer": Template("""Which organization is {{sub}}'s employer?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "member of sports team": Template("""Which sports team is {{sub}} a member of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "officeholder": Template("""Who holds the office of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "member of": Template("""Who or what is a member of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "educated at": Template("""Where was {{sub}} educated?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "chief executive officer": Template("""Who is the chief executive officer of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "director / manager": Template("""Who directs or manages {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "position held": Template("""What office does {{sub}} hold?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
}

prompt_QA_General_t = {
    "employer": Template("""In {{t}}, which organization is {{sub}}'s employer?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "member of sports team": Template("""In {{t}}, which sports team is {{sub}} a member of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "officeholder": Template("""In {{t}}, who holds the office of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "member of": Template("""In {{t}}, who or what is a member of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "educated at": Template("""In {{t}}, where was {{sub}} educated?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "chief executive officer": Template("""In {{t}}, who is the chief executive officer of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "director / manager": Template("""In {{t}}, who directs or manages {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "position held": Template("""In {{t}}, what office does {{sub}} hold?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
}

prompt_QA_Legal = Template("""Fill in the blank.
{{q}}
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")

prompt_QA_Legal_t = Template("""Fill in the blank.
In {{t}}, {{q}}
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")


prompt_QA_CommonSense = {
    "HasProperty": Template("""What properties does {{sub}} have?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotHasProperty": Template("""What properties does {{sub}} not have?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "CapableOf": Template("""What is {{sub}} capable of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotCapableOf": Template("""What is {{sub}} not capable of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Desires": Template("""What does {{sub}} desire?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotDesires": Template("""What does {{sub}} not desire?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Synonym": Template("""What is synonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Antonym": Template("""What is antonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}


prompt_QA_CommonSense_t = {
    "HasProperty": Template("""In {{t}}, What properties does {{sub}} have?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotHasProperty": Template("""In {{t}}, What properties does {{sub}} not have?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "CapableOf": Template("""In {{t}}, What is {{sub}} capable of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotCapableOf": Template("""In {{t}}, What is {{sub}} not capable of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Desires": Template("""In {{t}}, What does {{sub}} desire?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "NotDesires": Template("""In {{t}}, What does {{sub}} not desire?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Synonym": Template("""In {{t}}, What is synonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Antonym": Template("""In {{t}}, What is antonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}

prompt_QA_Math = {
    "Synonym": Template("""What is synonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Antonym": Template("""What is antonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Rely_on": Template("""What does {{sub}} rely on?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_rely_on": Template("""What relies on {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Belong_to": Template("""What does {{sub}} belong to?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_belong_to": Template("""What belongs to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Property_of": Template("""What is {{sub}} a property of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_property_of": Template("""What is propertie of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "None": Template("""Which of the following is unrelated to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Similar": Template("""What is similar to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Apposition": Template("""What is in apposition to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Other": Template("""What is related to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}

prompt_QA_Math_t = {
    "Synonym": Template("""In {{t}}, what is synonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Antonym": Template("""In {{t}}, what is antonym for {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Rely_on": Template("""In {{t}}, what does {{sub}} rely on?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_rely_on": Template("""In {{t}}, what relies on {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Belong_to": Template("""In {{t}}, what does {{sub}} belong to?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_belong_to": Template("""In {{t}}, what belongs to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Property_of": Template("""In {{t}}, what is {{sub}} a property of?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Inverse_property_of": Template("""In {{t}}, what is propertie of {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "None": Template("""In {{t}}, which of the following is unrelated to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Similar": Template("""In {{t}}, what is similar to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Apposition": Template("""In {{t}}, what is in apposition to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
"""),
    "Other": Template("""In {{t}}, what is related to {{sub}}?
(a) {{Ans1}} (b) {{Ans2}} (c) {{Ans3}} (d) {{Ans4}}
""")
}


### CK Sentence Form ###

Sent_Biomedical_t = {
    "isa": Template("""In {{t}}, {{sub}} is a {{obj}}."""),
    
    "inverse isa": Template("""In {{t}}, {{sub}} is an inverse of {{obj}}."""),
    
    "direct procedure site of": Template("""In {{t}}, {{sub}} is direct procedure site of {{obj}}."""),
    
    "indirect procedure site of": Template("""In {{t}}, {{sub}} is indirect procedure site of {{obj}}."""),
    
    "is primary anatomic site of disease": Template("""In {{t}}, {{sub}} is primary anatomic site of disease {{obj}}."""),
    
    "is not primary anatomic site of disease": Template("""In {{t}}, {{sub}} is not the primary anatomic site of disease {{obj}}."""),
    
    "is normal tissue origin of disease": Template("""In {{t}}, {{sub}} is normal tissue origin of disease {{obj}}."""),
    
    "is not normal tissue origin of disease": Template("""In {{t}}, {{sub}} is not normal tissue origin of disease {{obj}}."""),
    
    "is normal cell origin of disease": Template("""In {{t}}, {{sub}} is normal cell origin of disease {{obj}}."""),
    
    "is not normal cell origin of disease": Template("""In {{t}}, {{sub}} is not normal cell origin of disease {{obj}}."""),
    
    "has direct procedure site": Template("""In {{t}}, {{sub}} has direct procedure site {{obj}}."""),
    
    "has indirect procedure site": Template("""In {{t}}, {{sub}} has indirect procedure site {{obj}}."""),
    
    "positively regulates": Template("""In {{t}}, {{sub}} positively regulates {{obj}}."""),
    
    "negatively regulates": Template("""In {{t}}, {{sub}} negatively regulates {{obj}}.""")
}

single_prompt_QA_Biomedical_t = {
    "isa": Template("""In {{t}}, which of the following is classified as a {{sub}}?"""),
    "inverse isa": Template("""In {{t}}, which of the following is classified as an inverse of {{sub}}?"""),# reversed
    "direct procedure site of": Template("""In {{t}}, what is direct procedure site of {{sub}}?"""),
    "indirect procedure site of": Template("""In {{t}}, what is indirect procedure site of {{sub}}?"""),
    "is primary anatomic site of disease": Template("""In {{t}}, what is the primary anatomic site of {{sub}}?"""),
    "is not primary anatomic site of disease": Template("""In {{t}}, what is not the primary anatomic site of {{sub}}?"""),
    "is normal tissue origin of disease": Template("""In {{t}}, what is the normal tissue of origin for {{sub}}?"""),# reversed
    "is not normal tissue origin of disease": Template("""In {{t}}, what is not the normal tissue of origin for {{sub}}?"""),# reversed
    "is normal cell origin of disease": Template("""In {{t}}, what is the normal cell of origin for {{sub}}?"""),# reversed
    "is not normal cell origin of disease": Template("""In {{t}}, what is not the normal cell of origin for {{sub}}?"""),# reversed
    "has direct procedure site": Template("""In {{t}}, what has direct procedure site {{sub}}?"""),
    "has indirect procedure site": Template("""In {{t}}, what has indirect procedure site {{sub}}?"""),
    "positively regulates": Template("""In {{t}}, what positively regulates {{sub}}?"""),
    "negatively regulates": Template("""In {{t}}, what negatively regulates {{sub}}?"""),
}

single_prompt_QA_General_t = {
    "employer": Template("""In {{t}}, which organization is {{sub}}'s employer?"""),
    "member of sports team": Template("""In {{t}}, which sports team is {{sub}} a member of?"""),
    "officeholder": Template("""In {{t}}, who holds the office of {{sub}}?"""),
    "member of": Template("""In {{t}}, who or what is a member of {{sub}}?"""),
    "educated at": Template("""In {{t}}, where was {{sub}} educated?"""),
    "chief executive officer": Template("""In {{t}}, who is the chief executive officer of {{sub}}?"""),
    "director / manager": Template("""In {{t}}, who directs or manages {{sub}}?"""),
    "position held": Template("""In {{t}}, what office does {{sub}} hold?"""),
}

single_prompt_QA_Legal_t = Template("""Fill in the blank.
In {{t}}, {{q}}""")
