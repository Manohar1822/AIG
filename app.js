const { exec } = require('child_process');
const fs = require('fs');
require('dotenv').config();
const express=require("express");
const bodyParser=require("body-parser");
const mongoose=require("mongoose");

const ejs = require("ejs");

const session=require("express-session");
const passport=require("passport");
const passportLocalMongoose=require("passport-local-mongoose");
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const findOrCreate=require("mongoose-findOrCreate");
const nodemailer = require("nodemailer");

const app=express();

app.set("view engine","ejs");
app.use(bodyParser.urlencoded({extended:true}));
app.use(express.static("public"));

app.use(session({
    secret:"ManoharSecret",
    resave:false,
    saveUninitialized:false
}));
app.use(passport.initialize());
app.use(passport.session());

mongoose.connect("mongodb://127.0.0.1:27017/organizationsDB",() => {
    console.log("Mongo connected");
});

function sendOTP(email,auth,authcode){
    var smtpTransport = nodemailer.createTransport({
            host: 'smtp.gmail.com',
            port: 465,
            secure: true,
            auth: {
              user: process.env.EMAIL,
              pass: process.env.PASSWORD
            }
    });
    const link = `http://localhost:3000/linkverify/${authcode}`;
    var mailOptions;
    mailOptions={
        from: 'mmrchmanohar@gmail.com',
		to : email,
		subject : "Please confirm your Email account",
		html : "Your OTP for Email verification is: "+auth+"<br><br>Or,Click this link "+link+" <br>ThankYou <br>Regards,<br>AI-GURUKUL"	
	}
    smtpTransport.sendMail(mailOptions, function(error, response){
   	 if(error){
        	console.log(error);
	 }else{
        	console.log("Message sent");
    	 }
});


}



//function nextQuestion(examName,orgUsername,n){
    

//}



function saveCoToExam(examName,orgUsername,coNo){
    //console.log(examName);
    //console.log(orgUsername);
    //console.log(questionNo);
    CO.findOne({examName:examName,orgUsername:orgUsername,coNo:coNo},function(err,foundCo){
        if(foundCo){
            console.log("CO found");
            Exam.findOneAndUpdate({examName:examName,orgUsername:orgUsername},{ $push: { cos: foundCo  } },function(err,success){
                if(err){
                    console.log(err);
                }
                else if(success){
                    console.log(success);
                }
                else{
                    console.log("Nothing Happened");
                }
        })
        }
        else{
            console.log("CO not found");
        }
       
})
}



function saveToExam(examName,orgUsername,questionNo){
    //console.log(examName);
    //console.log(orgUsername);
    //console.log(questionNo);
    Question.findOne({examName:examName,orgUsername:orgUsername,questionNo:questionNo},function(err,foundQue){
        if(foundQue){
            console.log("Question found");
            Exam.findOneAndUpdate({examName:examName,orgUsername:orgUsername},{ $push: { questions: foundQue  } },function(err,success){
                if(err){
                    console.log(err);
                }
                else if(success){
                    console.log(success);
                }
                else{
                    console.log("Nothing Happened");
                }
        })
        }
        else{
            console.log("Question not found");
        }
       
})
}

function saveAnsToExam(examName,orgUsername,questionNo){
    Answer.findOne({examName:examName,orgUsername:orgUsername,questionNo:questionNo},function(err,foundQue){
        if(foundQue){
            console.log("Answer found");
            Exam.findOneAndUpdate({examName:examName,orgUsername:orgUsername},{ $push: { answers: foundQue  } },function(err,success){
                if(err){
                    console.log(err);
                }
                else if(success){
                    console.log(success);
                }
                else{
                    console.log("Nothing Happened");
                }
        })
        }
        else{
            console.log("Answer not found");
        }
       
})
}



function sendrequest(email,message){
    var smtpTransport = nodemailer.createTransport({
            host: 'smtp.gmail.com',
            port: 465,
            secure: true,
            auth: {
              user: process.env.EMAIL,
              pass: process.env.PASSWORD
            }
    });
    var mailOptions;
    mailOptions={
        from: 'mmrchmanohar@gmail.com',
		to : email,
		subject : "You have Join Request on AI-Gurukul app from your organization",
		html : ""+message+"<p>Thanks and regards</p><p>AI GuruKul</p>"	
	}
    smtpTransport.sendMail(mailOptions, function(error, response){
   	 if(error){
        	console.log(error);
	 }else{
        	console.log("Message sent");
    	 }
});


}

function randomStr(){
    let OTP="";
    const n=6;
    for(let i=0;i<n;i++){
        OTP=OTP+Math.floor(Math.random()*10);
    }
    return OTP;
}



function predictLevel(Ques){
  
    var level1=["Write", "List", "Label", "Name", "State", "Define", "Count", "Describe", "Draw", "Find", "Identify", "Match", 
                            "Quote", "Recall", "Recite", "Sequence", "Tell", "Arrange", "Duplicate", "Memorize", "Order", "Outline", 
                            "Recognize", "Relate", "Repeat", "Reproduce", "Select", "Choose", "Copy", "How", "Listen", "Locate",
                "Memorise", "Observe", "Omit", "Read", "Recognise", "Record", "Remember", "Retell", "Show", "Spell",
                "Trace", "What", "When", "Where", "Which", "Who", "Why"];
    var level2=["Explain", "Summarize", "Paraphrase", "Describe", "Illustrate", "Conclude", "Demonstrate", "Discuss",
                   "Generalize", "Identify", "Interpret", "Predict", "Report", "Restate", "Review", "Tell", "Classify",
                   "Convert", "Defend", "Distinguish", "Estimate", "Express", "Extend", "Give example", "Indicate",
                   "Infer", "Locate", "Recognize", "Rewrite", "Select", "Translate", "Ask", "Cite", "Compare",
                   "Contrast", "Generalise", "Give examples", "Match", "Observe", "Outline", "Purpose", "Relate",
                   "Rephrase", "Show", "Summarise","Comprehen"];
    var level3=["Use", "Compute", "Solve", "Demonstrate", "Apply", "Construct", "Change", "Choose", "Dramatize", "Interview",
                 "Prepare", "Produce", "Select", "Show", "Transfer", "Discover", "Employ", "Illustrate",
                 "Interpret", "Manipulate","Modify", "Operate", "Practice", "Predict", "Relate schedule", "Sketch",
                 "Use write", "Act", "Administer", "Associate", "Build", "Calculate", "Categorise", "Classify",
                 "Connect", "Correlation", "Develop", "Dramatise", "Experiment", "With", "Group", "Identify",
                 "Link", "Make use of", "Model", "Organise", "Perform", "Plan", "Relate", "Represent", "Simulate","Explain with","Explain along with","how",
                 "Summarise", "Teach", "Translate"];
    var level4=["Analyse", "Categorize", "Compare", "Contrast", "Separate", "Characterize", "Classify", "Debate", "Deduce", 
                "Diagram", "Differentiate", "Discriminate", "Distinguish", "Examine", "Outline", "Relate", "Research", 
                "Appraise", "Breakdown", "Calculate", "Criticize", "Derive", "Experiment", "Identify", "Illustrate", 
                "Infer", "Interpret", "Model", "Outline", "Point out", "Question", "Select", "Subdivide", "Test", 
                "Arrange", "Assumption", "Categorise", "Cause and", "Effect", "Choose", "Difference", "Discover", 
                "Dissect", "Distinction", "Divide", "Establish", "Find", "Focus", "Function", "Group", "Highlight", 
                "In-depth", "Discussion", "Inference", "Inspect", "Investigate", "Isolate", "List", "Motive", "Omit", 
                "Order", "Organise", "Point out", "Prioritize", "Rank", "Reason", "Relationships", "Reorganise", "See", 
                "Similar to", "Simplify", "Survey", "Take part in", "Test for", "Theme", "Comparing"];
    var level6=["Create", "Design", "Hypothesize", "Invent", "Develop", "Compose", "Construct", "Integrate", "Make",
                 "Organize", "Perform", "Plan", "Produce", "Propose", "Rewrite", "Arrange", "Assemble", "Categorize", 
                 "Collect", "Combine", "Comply", "Devise", "Explain", "Formulate", "Generate", "Prepare", "Rearrange",
                 "Reconstruct", "Relate", "Reorganize", "Revise", "Set up", "Summarize", "Synthesize", "Tell", "Write", 
                 "Adapt", "Add to", "Build", "Change", "Choose", "Combine", "Compile", "Convert", "Delete", "Discover", 
                 "Discuss", "Elaborate", "Estimate", "Experiment", "Extend", "Happen", "Hypothesise", "Imagine",
                 "Improve", "Innovate", "Make up", "Maximise", "Minimise", "Model", "Modify", "Original", "Originate",
                 "Predict", "Reframe", "Simplify", "Solve", "Speculate", "Substitute", "Suppose", "Tabulate", "Test", 
                 "Theorise", "Think", "Transform", "Visualise","specify"];
    var level5=["Judge", "Recommend", "Critique", "Justify", "Appraise", "Argue", "Assess", "Choose", "Conclude", 
                "Decide", "Evaluate", "Predict", "Prioritize", "Prove", "Rank", "Rate", "Select", "Attach", "Compare", 
                "Contrast", "Defend", "Describe", "Discriminate", "Estimate", "Explain", "Interpret", "Relate",
                "Summarize", "Support", "Value", "Agree", "Award", "Bad", "Consider", "Convince", "Criteria", 
                "Criticise", "Debate", "Deduct", "Determine", "Disprove", "Dispute", "Effective", "Give reasons", "Good",
                "Grade", "How do we", "Know", "Importance", "Infer", "Influence", "Mark", "Measure", "Opinion", 
                "Perceive", "Persuade", "Prioritise", "Rule on", "Test", "Useful", "Validate", "Why"];
    
    
  
    var que=Ques.toString().toLowerCase();
    var c=[0,0,0,0,0,0];
    level1.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[0]++;
        }
    })
    
    level2.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[1]++;
        }
    })
    
    level3.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[2]++;
        }
    })
    
    level4.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[3]++;
        }
    })
    
    level5.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[4]++;
        }
    })
    
    level6.forEach(function(wor){
        var word=wor.toString().toLowerCase();
        if(que.indexOf(word)!=-1){
            c[5]++;
        }
    })
    
    var maxi=0;
    var ans=1;
    for(var i=0;i<c.length;i++){
        if(maxi<c[i]){
            maxi=c[i]
            ans=i+1;
        }
    }
    
    console.log("It is level ",ans)
    return ans;
    
    }




const schemaOptions = {
    timestamps: { createdAt: 'created_at', updatedAt: 'updated_at' },
  };


const organizationSchema=new mongoose.Schema({
    username:{type:String,sparse:true,unique:false},
    password:String,
    googleId:String,
    name:String,
    auth:String,
    role:String,
    pincode:String,
    class:String,
    orgName:String,
    orgUsername:String,
    orgID:String,
    orgTypeOfID:String,
    phone:String,
    optionalPhone:String,
    approved:{
        type:Boolean,
        default:false
    },
    registeredTeacher:{
        type:Boolean,
        default:false
    },
    registeredStudent:{
        type:Boolean,
        default:false
    },
    registeredOrg:{
        type:Boolean,
        default:false
    }
},schemaOptions);
const classSchema=new mongoose.Schema({
    className:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    classStudent:{type:[organizationSchema],index:false,sparse:false,unique:false},
    classTeacher:{type:[organizationSchema],index:false,sparse:false,unique:false},
    classOrg:{type:[organizationSchema],index:false,sparse:false,unique:false}
});

const questionSchema=new mongoose.Schema({
    creatorUsername:{type:String,sparse:true,unique:false},
    questionNo:{type:String,sparse:true,unique:false},
    examName:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    question:{type:String,sparse:true,unique:false},
    solution:{type:String,sparse:true,unique:false},
    mainPoints:{type:[String],sparse:true,unique:false},
    keyWords:{type:[String],sparse:true,unique:false},
    maxMarks:{
        type: Number,
        sparse:true,
        unique:false,
        integer: true
    },
    coNo:{type:String,sparse:true,unique:false},
    co:{type:String,sparse:true,unique:false},
    level:{type:String,sparse:true,unique:false}
})

const answerSchema=new mongoose.Schema({
    studentUsername:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    examName:{type:String,sparse:true,unique:false},
    questionNo:{type:String,sparse:true,unique:false},
    question:{type:questionSchema,sparse:true,unique:false},
    answerNo:{type:String,sparse:true,unique:false},
    answer:{type:String,sparse:true,unique:false},
    obtainMarks:{
        type: Number,
        sparse:true,
        unique:false,
        integer:true,
        default:0
    }
})

const gradeSchema=new mongoose.Schema({
    studentUsername:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    examName:{type:String,sparse:true,unique:false},
    answerNo:{type:String,sparse:true,unique:false},
    answer:{type:answerSchema,sparse:true,unique:false},
    obtainMarks:{
        type: Number,
        sparse:true,
        unique:false,
        integer:true,
        default:0
    }
})

const coSchema=new mongoose.Schema({
    creatorUsername:{type:String,sparse:true,unique:false},
    examName:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    co:{type:String,sparse:true,unique:false},
    coNo:{type:String,sparse:true,unique:false}
})

const examSchema=new mongoose.Schema({
    examName:{type:String,sparse:true,unique:false},
    className:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    class:{type:[classSchema],sparse:true,unique:false},
    examDate:{type:Date,sparse:true,unique:false},
    examEndDate:{type:Date,sparse:true,unique:false},
    questions:{type:[questionSchema],sparse:true,unique:false},
    answers:{type:[answerSchema],sparse:true,unique:false},
    cos:{type:[coSchema],sparse:true,unique:false},
    examGrade:{type: Number,
        sparse:true,
        unique:false,
        integer: true
    },
    gradeSchema:{type:[gradeSchema],sparse:true,unique:false}
});



organizationSchema.plugin(passportLocalMongoose,{usernameField: 'username'});
organizationSchema.plugin(findOrCreate);





const Organization=new mongoose.model("User",organizationSchema);
const Class=new mongoose.model("Class",classSchema);
const Question=new mongoose.model("Question",questionSchema);
const Answer=new mongoose.model("Answer",answerSchema);
const Grade=new mongoose.model("Grade",gradeSchema);
const Exam=new mongoose.model("Exam",examSchema);
const CO=new mongoose.model("CO",coSchema);

passport.use(Organization.createStrategy());
passport.serializeUser(function(user,done){
    done(null,user.id);
});
passport.deserializeUser(function(id,done){
    Organization.findById(id,function(err,user){
        done(err,user);
    });
});



/*passport.use(new GoogleStrategy({
    clientID: process.env.CLIENT_ID,
    clientSecret: process.env.CLIENT_SECRET,
    callbackURL: "http://localhost:3000/auth/google/organizationpage",
    userProfileURL:"https://www.googleapis.com/oauth2/v3/userinfo"
  },
  function(accessToken, refreshToken, profile, cb) {
    //console.log(profile);
    //console.log(accessToken);
    //console.log(refreshToken);
    //console.log(profile.emails[0].value);Not Working
    Organization.findOrCreate({ googleId: profile.id }, function (err, user) {
        //user.username:profile.emails[0];
        //user.save();
      return cb(err, user);
    });
  }
));

app.get("/auth/google",
  passport.authenticate("google", { scope: ['profile'] }));

  app.get("/auth/google/organizationpage", 
  passport.authenticate("google", { failureRedirect: "/login" }),
  function(req, res) {
    // Successful authentication, redirect home.
    req.user.registeredOrg=true;
    req.user.save();
    res.redirect('/organizationpage');
  });
*/

app.get("/",function(req,res){
    res.render("first-page");
});

app.get("/loginorganization",function(req,res){
    res.render("loginOrg",{message:""});
});
app.get("/registerorganization",function(req,res){
    res.render("registerOrg",{message:""});
});

app.post("/registerorganization",function(req,res){
    const username=req.body.username;
    const pincode=req.body.pincode;
    console.log("In the post /registerorganization");
    Organization.register({username:req.body.username},req.body.password,function(err,user){
        if(err){
            console.log(err);
            res.render("registerOrg",{message:"User Already Exists, Please try to login with same username"});
        }
        else{
            
            passport.authenticate("local")(req,res,function(){
                Organization.findOne({username:username},function(err,foundUser){
                    foundUser.role="Admin";
                    foundUser.orgUsername=foundUser.username;
                    foundUser.save();
                });
                res.redirect("/sendemailOrg");
            });
            
        }
        
    });
});

app.post("/loginorganization",function(req,res){
    const user=new Organization({
        username:req.body.username,
        password:req.body.password
    });
    Organization.findOne({username:req.body.username},function(err,foundUser){
        if(!foundUser){
            res.render("loginOrg",{message:"User Not Found, Please try to Register with same username"})
        }
        else{
            req.login(user,function(err){
                if(err){
                    console.log("User Not Found");
                    console.log(err);
                }
                else{
                    passport.authenticate("local")(req,res,function(){
                        res.redirect("/homeorganization");
                    });
                }
            });
        }
    })
    
})


app.post("/nameorganization",function(req,res){
    //console.log(req.user);
    console.log("in post/nameorganization");
    Organization.findOneAndUpdate({_id:req.user._id},{$set:{name:req.body.OrganizationName,pincode:req.body.pincode}},function(err,foundOrganization){
        if(!err){
            res.redirect("/homeorganization");
        }
    })
    
})

app.get("/organizationpage",function(req,res){
    if(req.isAuthenticated()){
        //console.log(req.user.name);
        if(req.user.name){
            res.redirect("/homeorganization");
        }
        else{
            res.render("organizationpage");
        }
        
    }
    else{
        res.redirect("/loginorganization");
    }
});

app.get("/homeorganization",function(req,res){
    //console.log("Under /homeorganization and data of user is "+req.user);
    if(req.isAuthenticated()){
        if(req.user.registeredOrg){
            const orgUsername=req.user.username;
            Exam.find({orgUsername:orgUsername},function(err,foundExam){
                if(foundExam){
                    Class.find({"classOrg.username":orgUsername},function(err,foundClass){
                
                        if(foundClass){
                            Organization.find({"orgUsername":orgUsername,role:"Teacher",registeredTeacher:true},function(err,foundTeacher){
                                if(foundTeacher){
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:foundTeacher,Student:foundStudent,Exam:foundExam});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:foundTeacher,Student:false,Exam:foundExam});
                                        }
                                    })
                                }
                                else{
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:false,Student:foundStudent,Exam:foundExam});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:false,Student:false,Exam:foundExam});
                                        }
                                    })
                                }
                            })
                        }
                        else{
                            Organization.find({"orgUsername":orgUsername,role:"Teacher",registeredTeacher:true},function(err,foundTeacher){
                                if(foundTeacher){
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:foundTeacher,Student:foundStudent,Exam:foundExam});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:foundTeacher,Student:false,Exam:foundExam});
                                        }
                                    })
                                }
                                else{
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:false,Student:foundStudent,Exam:foundExam});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:false,Student:false,Exam:foundExam});
                                        }
                                    })
                                }
                            })
                        }
                    })
                }
                else{
                    Class.find({"classOrg.username":orgUsername},function(err,foundClass){
                
                        if(foundClass){
                            Organization.find({"orgUsername":orgUsername,role:"Teacher",registeredTeacher:true},function(err,foundTeacher){
                                if(foundTeacher){
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:foundTeacher,Student:foundStudent,Exam:false});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:foundTeacher,Student:false,Exam:false});
                                        }
                                    })
                                }
                                else{
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:false,Student:foundStudent,Exam:false});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:foundClass,Teacher:false,Student:false,Exam:false});
                                        }
                                    })
                                }
                            })
                        }
                        else{
                            Organization.find({"orgUsername":orgUsername,role:"Teacher",registeredTeacher:true},function(err,foundTeacher){
                                if(foundTeacher){
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:foundTeacher,Student:foundStudent,Exam:false});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:foundTeacher,Student:false,Exam:false});
                                        }
                                    })
                                }
                                else{
                                    Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                                        if(foundStudent){
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:false,Student:foundStudent,Exam:false});
                                        }
                                        else{
                                            res.render("homeorganization",{User:req.user,Clas:false,Teacher:false,Student:false,Exam:false});
                                        }
                                    })
                                }
                            })
                        }
                    })
                }
            });
        }
        else{
            Organization.findOne({username:req.user.username,role:"Admin"},function(err,foundUser){
                //Organization.findByIdAndRemove(foundUser._id,function(err){
                    if(foundUser){
                        console.log("Verify your email First");
                        if(foundUser.role=="Admin"){
                            res.redirect("/sendemailOrg");
                        }
                        else{
                            res.render("unauthorized");
                        }
                    }
                    else{
                        res.render("unauthorized");
                    }
                //})
            })
        }
    }
    else{
        res.redirect("/loginorganization");
    }
})

app.get("/logoutorganization",function(req,res){
    req.logout(function(err){
        if(err){
            res.send(err);
        }
    });
    res.redirect("/");
});

app.get("/sendemailOrg",function(req,res){
    if(req.isAuthenticated()){
        let rand=randomStr();
    const auth=req.user.username+rand;
    req.user.auth=auth;
    req.user.save();
    //console.log(req.user.auth);
    sendOTP(req.user.username,rand,auth);
    res.render("enterotp",{email:req.user.username,message:""});
    }
    else{
        res.redirect("/");
    }
});

app.post("/verifyOrg",function(req,res){
    
        const OTP=req.body.OTP;
    const enteredAuth=req.user.username+OTP;
    if(enteredAuth==req.user.auth){
        if(req.user.role=="Admin"){
            req.user.registeredOrg=true;
            req.user.save();
        }
        res.redirect("/organizationpage");
    }
    else{
        console.log("Incorrect OTP");
        res.render("enterotp",{email:req.user.username, message:"Incorrect OTP, enter the correct OTP"});
    }
    
    
});


app.get("/deleteOrganization",function(req,res){
    Organization.findOne({username:req.user.username},function(err,foundUser){
        Organization.findByIdAndRemove(foundUser._id,function(err){
            if(!err){
                console.log("Deleted Successfully");
                res.redirect("/");
            }
        })
    })
});

app.get("/linkverify/:auth",function(req,res){
    const reqAuth=req.params.auth;
    Organization.findOne({auth:reqAuth},function(err,foundUser){
        if(!err){
            if(foundUser.role=="Admin"){
                foundUser.registeredOrg=true;
        foundUser.save();
        res.redirect("/organizationpage");
            }
            else if(foundUser.role=="Student"){
                foundUser.registeredStudent=true;
        foundUser.save();
        res.redirect("/studentpage1");
            }
            else if(foundUser.role=="Teacher"){
                foundUser.registeredTeacher=true;
        foundUser.save();
        res.redirect("/teacherpage1");
            }
            
        }
        else{
            console.log("User Not Found");
            res.redirect("/")
        }
    })

});

app.get("/aboutus",function(req,res){
    res.render("aboutUs");
})

app.get("/adminfirstpage",function(req,res){
    res.render("adminFirstPage");
})

app.get("/teacherfirstpage",function(req,res){
    res.render("teacherFirstPage");
})

app.get("/studentfirstpage",function(req,res){
    res.render("studentFirstPage");
})


app.get("/loginchoose",function(req,res){
    res.render("loginChooseUser");
})

app.get("/registerchoose",function(req,res){
    res.render("registerChooseUser");
})

app.get("/loginteacher",function(req,res){
    res.render("loginteacher",{message:""});
});

app.get("/loginstudent",function(req,res){
    res.render("loginstudent",{message:""});
});

app.get("/registerteacher",function(req,res){
    res.render("registerTeacher",{message:""});
})

app.get("/registerstudent",function(req,res){
    Organization.find({},function(err,foundOrgs){
        if(!err){
            res.render("registerStudent",{message:"", organization:foundOrgs});
        }
        else{
            res.render("registerStudent",{message:"", organization:false});
        }
    });
    

})

app.get("/createClass",function(req,res){
    if(req.isAuthenticated()){
        if(req.user.registeredOrg){
            res.render("createClass",{name:req.user.name,message:""});
        }
        else{
            Organization.findOne({username:req.user.username,role:"Admin"},function(err,foundUser){
                //Organization.findByIdAndRemove(foundUser._id,function(err){
                    if(!err){
                        console.log("Verify your email First");
                        res.redirect("/sendemailOrg");
                    }
                    else{
                        res.render("unauthorized");
                    }
                //})
            })
        }
    }
    else{
        res.redirect("/loginorganization");
    }
});

app.post("/createClass",function(req,res){
    console.log("post createClass");
    Class.findOne({className:req.body.name, orgUsername:req.user.username},function(err,foundClass){
        if(foundClass){
            res.render("createClass",{message:"Class Already Exist"})
        }
        else{
            const newclass=new Class({
                className:req.body.name,
                orgUsername:req.user.username,
                classOrg:req.user
            });
            newclass.save();
            /*Class.findOne({name:req.body.name, orgUsername:req.user.username},function(err,class){
                class.classOrg.push({})
            })*/
                res.redirect("/homeorganization");
            
            //req.user.classes.push(newclass);
            //req.user.save();
        }
    })
    
});


/*











STUDENT PART











*/


app.post("/registerstudent",function(req,res){
    const username=req.body.username;
    const pincode=req.body.pincode;
    console.log("In the post /registerstudent");
    Organization.register({username:req.body.username},req.body.password,function(err,user){
        if(err){
            console.log(err);
            res.render("registerStudent",{message:"User Already Exists, Please try to login with same username"});
        }
        else{
            
            passport.authenticate("local")(req,res,function(){
                Organization.findOne({username:username},function(err,foundUser){
                    foundUser.role="Student";
                    foundUser.save();
                });
                res.redirect("/sendemailStudent");
            });
            
        }
        
    });
})

app.get("/sendemailStudent",function(req,res){
    if(req.isAuthenticated()){
        let rand=randomStr();
    const auth=req.user.username+rand;
    req.user.auth=auth;
    req.user.save();
    //console.log(req.user.auth);
    sendOTP(req.user.username,rand,auth);
    res.render("enterotpStudent",{email:req.user.username,message:""});
    }
    else{
        res.redirect("/");
    }
    
});

app.post("/verifyStudent",function(req,res){
    const OTP=req.body.OTP;
    const enteredAuth=req.user.username+OTP;
    if(enteredAuth==req.user.auth){
        if(req.user.role=="Student"){
            req.user.registeredStudent=true;
            req.user.save();
        }
        //req.user.role="Admin";
        res.redirect("/studentpage1");
    }
    else{
        console.log("Incorrect OTP");
        res.render("enterotpStudent",{email:req.user.username, message:"Incorrect OTP, enter the correct OTP"});
    }
})

app.get("/studentpage1",function(req,res){
    if(req.isAuthenticated() && req.user.role=="Student"){
        //console.log(req.user.name);
        if(req.user.name){

            res.redirect("/homestudent");
        }
        else{
            Organization.find({role:"Admin",registeredOrg:true,name:{"$ne":""}},function(err,foundOrg){
                if(!err){
                    res.render("studentpage1",{organizations:foundOrg});
                }
                else{
                    res.render("studentpage1",{organizations:false});   
                }
            })
            
        }
        
    }
    else{
        res.redirect("/loginstudent");
    }
});

app.post("/orgstudent",function(req,res){
    if(req.isAuthenticated() && req.user.role=="Student"){
        if(req.body.orgUsername){
            req.user.orgUsername=req.body.orgUsername;
            
            Organization.findOne({username:req.body.orgUsername},function(err,foundOrg){
                if(!err){
                    req.user.orgName=foundOrg.name;
                    req.user.save();
                }
            })
            
        
        Class.find({orgUsername:req.body.orgUsername},function(err,foundClass){
            if(!err){
                res.render("studentpage2",{classes:foundClass});
            }
            else{
                res.render("studentpage2",{classes:false});
            }
            
        })
        }
        else{
            res.redirect("/studentpage1");
        }
        
    }
    else{
        res.redirect("/");
    }
});

app.post("/detailstudent",function(req,res){
    if(req.isAuthenticated() && req.user.orgUsername && req.user.role=="Student"){
        req.user.name=req.body.name;
        req.user.class=req.body.class;
        req.user.orgTypeOfID=req.body.typeOfID;
        req.user.orgID=req.body.orgID;
        req.user.phone=req.body.phone;
        req.user.optionalPhone=req.body.optionalPhone;
        req.user.save();
        /*Organization.findOne({username:req.user.orgUsername}, function(err,foundOrg){
            req.user.orgName=foundOrg.name;
            req.user.save();
        })*/
        Class.findOne({className:req.body.class,orgUsername:req.user.orgUsername},function(err,foundClass){
            foundClass.classStudent.push(req.user);
            foundClass.save();
        })
        res.redirect("/homestudent");
    }
    else{
        res.redirect("/studentpage1");
    }
});

app.get("/homestudent",function(req,res){
    if(req.isAuthenticated()){
        if(req.user.registeredStudent){
            if(req.user.name){
                const orgUsername=req.user.orgUsername;
                Class.findOne({"classStudent.username":req.user.username},function(err,foundClass){
                    if(foundClass){
                        Exam.find({orgUsername:orgUsername,className:req.user.class},function(err,foundExam){
                            if(!err){
                                res.render("homestudent",{User:req.user,Clas:foundClass,Exam:foundExam});
                            }
                            else{
                                res.render("homestudent",{User:req.user,Clas:foundClass,Exam:false});
                            }
                        });
                    }
                    else{
                        Exam.find({orgUsername:orgUsername,className:req.user.class},function(err,foundExam){
                            if(!err){
                                res.render("homestudent",{User:req.user,Clas:false,Exam:foundExam});
                            }
                            else{
                                res.render("homestudent",{User:req.user,Clas:false,Exam:false});
                            }
                        });
                    }
                })
                
            }
            else{
                res.redirect("/studentpage1");
            }
            
        }
        else{
            Organization.findOne({username:req.user.username,role:"Student"},function(err,foundUser){
                //Organization.findByIdAndRemove(foundUser._id,function(err){
                    if(foundUser){
                        console.log("Verify your email First");
                        if(foundUser.role=="Student"){
                            res.redirect("/sendemailStudent");
                        }
                        else{
                            res.render("unauthorized");
                        }
                    }
                    else{
                        res.render("unauthorized");
                    }
                //})
            })
        }
    }
    else{
        res.redirect("/loginstudent");
    }
});

app.post("/loginstudent",function(req,res){
    const user=new Organization({
        username:req.body.username,
        password:req.body.password
    });
    Organization.findOne({username:req.body.username},function(err,foundUser){
        if(!foundUser){
            res.render("loginstudent",{message:"User Not Found, Please try to Register with same username"})
        }
        else{
            req.login(user,function(err){
                if(err){
                    console.log("User Not Found");
                    console.log(err);
                }
                else{
                    passport.authenticate("local")(req,res,function(){
                        res.redirect("/homestudent");
                    });
                }
            });
        }
    })
})
/*







TEACHER PART










*/
app.post("/registerteacher",function(req,res){
    const username=req.body.username;
    const pincode=req.body.pincode;
    console.log("In the post /registerteacher");
    Organization.register({username:req.body.username},req.body.password,function(err,user){
        if(err){
            console.log(err);
            res.render("registerTeacher",{message:"User Already Exists, Please try to login with same username"});
        }
        else{
            
            passport.authenticate("local")(req,res,function(){
                Organization.findOne({username:username},function(err,foundUser){
                    foundUser.role="Teacher";
                    foundUser.save();
                });
                res.redirect("/sendemailTeacher");
            });
            
        }
        
    });
});

app.get("/sendemailTeacher",function(req,res){
    if(req.isAuthenticated()){
        let rand=randomStr();
    const auth=req.user.username+rand;
    req.user.auth=auth;
    req.user.save();
    //console.log(req.user.auth);
    sendOTP(req.user.username,rand,auth);
    res.render("enterotpTeacher",{email:req.user.username,message:""});
    }
    else{
        res.redirect("/");
    }
});

app.post("/verifyTeacher",function(req,res){
    const OTP=req.body.OTP;
    const enteredAuth=req.user.username+OTP;
    if(enteredAuth==req.user.auth){
        if(req.user.role=="Teacher"){
            req.user.registeredTeacher=true;
            req.user.save();
        }
        
        res.redirect("/teacherpage1");
    }
    else{
        console.log("Incorrect OTP");
        res.render("enterotpTeacher",{email:req.user.username, message:"Incorrect OTP, Please enter the correct OTP"});
    }
});


app.get("/teacherpage1",function(req,res){
    if(req.isAuthenticated() && req.user.role=="Teacher"){
        //console.log(req.user.name);
        if(req.user.name){

            res.redirect("/hometeacher");
        }
        else{
            Organization.find({role:"Admin",registeredOrg:true,name:{"$ne":""}},function(err,foundOrg){
                if(!err){
                    res.render("teacherpage1",{organizations:foundOrg});
                }
                else{
                    res.render("teachertpage1",{organizations:false});   
                }
            })
            
        }
        
    }
    else{
        res.redirect("/loginteacher");
    }
});


app.post("/orgteacher",function(req,res){
    if(req.isAuthenticated() && req.user.role=="Teacher"){
        if(req.body.orgUsername){
            req.user.orgUsername=req.body.orgUsername;
            
            Organization.findOne({username:req.body.orgUsername},function(err,foundOrg){
                if(!err){
                    req.user.orgName=foundOrg.name;
                    req.user.save();
                }
            })
            
        
        Class.find({orgUsername:req.body.orgUsername},function(err,foundClass){
            if(!err){
                res.render("teacherpage2",{classes:foundClass});
            }
            else{
                res.render("teacherpage2",{classes:false});
            }
            
        })
        }
        else{
            res.redirect("/teacherpage1");
        }
        
    }
    else{
        res.redirect("/");
    }
});


app.post("/detailteacher",function(req,res){
    if(req.isAuthenticated() && req.user.orgUsername && req.user.role=="Teacher"){
        req.user.name=req.body.name;
        req.user.class=req.body.class;
        req.user.orgTypeOfID=req.body.typeOfID;
        req.user.orgID=req.body.orgID;
        req.user.phone=req.body.phone;
        req.user.optionalPhone=req.body.optionalPhone;
        req.user.save();
        /*Organization.findOne({username:req.user.orgUsername}, function(err,foundOrg){
            req.user.orgName=foundOrg.name;
            req.user.save();
        })*/
        Class.find({orgUsername:req.user.orgUsername},function(err,foundClasses){
            foundClasses.forEach(function(clas){
                clas.classTeacher.push(req.user);
                clas.save();
            })
        })
        res.redirect("/hometeacher");
    }
    else{
        res.redirect("/teacherpage1");
    }
});


app.get("/hometeacher",function(req,res){
    
    if(req.isAuthenticated()){
        const orgUsername=req.user.orgUsername;
        if(req.user.registeredTeacher && req.user.approved){
            if(req.user.name){
                Class.find({"classTeacher.username":req.user.username},function(err,foundClass){
                    if(foundClass){
                        Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                            if(foundStudent){
                                Exam.find({orgUsername:orgUsername},function(err,foundExam){
                                    if(foundExam){
                                        res.render("hometeacher",{User:req.user,Clas:foundClass,Student:foundStudent,Exam:foundExam});
                                    }
                                    else{
                                        res.render("hometeacher",{User:req.user,Clas:foundClass,Student:foundStudent,Exam:false});

                                    }
                                })
                                
                            }
                            else{
                                Exam.find({orgUsername:orgUsername},function(err,foundExam){
                                    if(foundExam){
                                        res.render("hometeacher",{User:req.user,Clas:foundClass,Student:false,Exam:foundExam});
                                    }
                                    else{
                                        res.render("hometeacher",{User:req.user,Clas:foundClass,Student:false,Exam:false});

                                    }
                                })
                            }
                        });
                    }
                    else{
                        Organization.find({orgUsername:orgUsername,role:"Student",registeredStudent:true},function(err,foundStudent){
                            if(foundStudent){
                                Exam.find({orgUsername:orgUsername},function(err,foundExam){
                                    if(foundExam){
                                        res.render("hometeacher",{User:req.user,Clas:false,Student:foundStudent,Exam:foundExam});
                                    }
                                    else{
                                        res.render("hometeacher",{User:req.user,Clas:false,Student:foundStudent,Exam:false});

                                    }
                                })
                                
                            }
                            else{
                                Exam.find({orgUsername:orgUsername},function(err,foundExam){
                                    if(foundExam){
                                        res.render("hometeacher",{User:req.user,Clas:false,Student:false,Exam:foundExam});
                                    }
                                    else{
                                        res.render("hometeacher",{User:req.user,Clas:false,Student:false,Exam:false});

                                    }
                                })
                            }
                        });
                    }
                })
                
            }
            else{
                res.redirect("/teacherpage1");
            }
        }
        else if(req.user.registeredTeacher){
            res.render("notApproved");
        }
        else{
            Organization.findOne({username:req.user.username,role:"Teacher"},function(err,foundUser){
                //Organization.findByIdAndRemove(foundUser._id,function(err){
                    if(foundUser){
                        console.log("Verify your email First");
                        if(foundUser.role=="Teacher"){
                            res.redirect("/sendemailTeacher");
                        }
                        else{
                            res.render("unauthorized");
                        }
                    }
                    else{
                        res.render("unauthorized");
                    }
                //})
            })
        }
    }
    else{
        res.redirect("/loginteacher");
    }
});


app.post("/loginteacher",function(req,res){
    const user=new Organization({
        username:req.body.username,
        password:req.body.password
    });
    Organization.findOne({username:req.body.username},function(err,foundUser){
        if(!foundUser){
            res.render("loginteacher",{message:"User Not Found, Please try to Register with same username"})
        }
        else{
            req.login(user,function(err){
                if(err){
                    console.log("User Not Found");
                    console.log(err);
                }
                else{
                    passport.authenticate("local")(req,res,function(){
                        res.redirect("/hometeacher");
                    });
                }
            });
        }
    })
})

app.get("/addteacher",function(req,res){
    if(req.isAuthenticated()){
        if(req.user.role=="Admin"&&req.user.registeredOrg==true){
            res.render("addteacher",{Org:req.user});
        }
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }
});

app.post("/addteacher",function(req,res){
    const email=req.body.email;
    const message="Hello,\nGreetings of the day,\nYou are invited by organization: "+req.user.name+","+req.user.pincode+"\n Please visit our AI Gurukul App now to join this organization as teacher"
    sendrequest(email,message);
    res.redirect("/homeorganization")
})

/*

















*/


app.post("/approvedelete",function(req,res){
    const id=req.body.userID;
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){
            //console.log("Inside");
            if(req.body.Request=="Approve"){
                //console.log("Approve Pending");
                Organization.findOne({_id:id},function(err,foundUser){
                    foundUser.approved=true;
                    foundUser.save();
                    if(req.user.role=="Admin"){
                        res.redirect("/homeorganization");
                    }
                    else{
                        res.redirect("/hometeacher");
                    }
                    
                })
            }
            else if(req.body.Request=="Remove"){
                Organization.findByIdAndRemove(id,function(err){
                    if(!err){
                        console.log("Successfully Removed and deleted account");
                        if(req.user.role=="Admin"){
                            res.redirect("/homeorganization");
                        }
                        else{
                            res.redirect("/hometeacher");
                        }
                    }
                });
            }
            else if(req.body.Request=="Reject"){
                Organization.findByIdAndRemove(id,function(err){
                    if(!err){
                        console.log("Successfully Removed and deleted account");
                        if(req.user.role=="Admin"){
                            res.redirect("/homeorganization");
                        }
                        else{
                            res.redirect("/hometeacher");
                        }
                    }
                });
            }
        }
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }
    
})


app.get("/addstudent",function(req,res){
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){
            res.render("addstudent",{Org:req.user});
        }
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }
});



app.post("/addstudent",function(req,res){
    const email=req.body.email;
    const message="Hello,\nGreetings of the day,\nYou are invited by organization: "+req.user.name+","+req.user.pincode+"\n Please visit our AI Gurukul App now to join this organization as a Student"
    sendrequest(email,message);
    res.redirect("/homeorganization")
})
/*























*/

app.get("/createExam",function(req,res){
    //console.log(req.user);
    const orgUsername=req.user.orgUsername;
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){
            Class.find({orgUsername:req.user.orgUsername},function(err,foundClass){
                if(foundClass){
                    res.render("createExam",{Org:req.user,message:"",Classes:foundClass});
                }
                else{
                    res.render("createExam",{Org:req.user,message:"",Classes:false});
                }
            })
        }
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }
})


app.post("/createExam",function(req,res){
    Exam.findOne({examName:req.body.examName, orgUsername:req.user.orgUsername},function(err,foundExam){
        if(foundExam){
            Class.find({orgUsername:req.user.orgUsername},function(err,foundClass){
                if(foundClass){
                    res.render("createExam",{Org:req.user,message:"Exam Already Exist",Classes:foundClass});
                }
                else{
                    res.render("createExam",{Org:req.user,message:"Exam Already Exist",Classes:false});
                }
            })
        }
        else{
    Class.findOne({_id:req.body.classId},function(err,foundClass){
        const exam=new Exam({
            class:foundClass,
            className:foundClass.className,
            examName:req.body.examName,
            examDate:req.body.examDate,
            examEndDate:req.body.examEndDate,
            orgUsername:req.user.orgUsername
        })
        exam.save();
        if(req.user.role=="Admin"){
            res.redirect("/homeorganization");
        }
        else{
            res.redirect("/hometeacher");
        }
    })
}

})
});

app.post("/removeExam",function(req,res){
    const examName=req.body.examName;
    const orgUsername=req.user.orgUsername;
    Exam.deleteMany({examName:examName,orgUsername:orgUsername},function(err,foundExam){
        if(err){
            console.log(err);
        }
    })
    if(req.user.role=="Admin"){
        res.redirect("/homeorganization");
    }
    else{
        res.redirect("/hometeacher");
    }
    
});


app.post("/getaddQuestion",function(req,res){
    const examName=req.body.examName;
    //console.log("examName= "+examName);
    const orgUsername=req.user.orgUsername;
    //console.log("orgUsername= "+orgUsername);
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){

            Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                if(foundExam){
                    Question.find({examName:examName,orgUsername:orgUsername},function(err,foundQuestion){
                        if(foundQuestion){
                            //console.log(foundExam);
                            CO.find({examName:examName,orgUsername:orgUsername},function(err,foundCo){
                                if(foundCo){
                                    res.render("addQuestion",{User:req.user,Exams:foundExam,Question:foundQuestion,Co:foundCo,message:""});
                                }
                                else{
                                    res.render("addQuestion",{User:req.user,Exams:foundExam,Question:foundQuestion,Co:false,message:""});
                                }
                            })
                        }
                        else{
                            CO.find({examName:examName,orgUsername:orgUsername},function(err,foundCo){
                                if(foundCo){
                                    res.render("addQuestion",{User:req.user,Exams:foundExam,Question:false,Co:foundCo,message:""});
                                }
                                else{
                                    res.render("addQuestion",{User:req.user,Exams:foundExam,Question:false,Co:false,message:""});
                                }
                            })
                        }
                    })
                
                }
                else{
                    console.log("No exam found with this in app.get(/addQuestion)");
                    res.redirect("/back");
                }
                
            })
            
        }
        
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }

});





app.post("/addQuestion",function(req,res){
    const examName=req.body.examName;
    const coNo=req.body.coNo;
    const level=predictLevel(req.body.question);
    var cos="";
    //console.log(examName);
    const orgUsername=req.user.orgUsername;
    Question.findOne({examName:examName,orgUsername:orgUsername,questionNo:req.body.questionNo},function(err,foundQuestion){
        
        if(foundQuestion){
            Exam.find({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                res.render("addQuestion",{User:req.user,Exams:foundExam,Question:foundQuestion,message:"Question Number Already Exist"});
            })
            
        }
        else{
            
            
            const question=new Question({
                questionNo:req.body.questionNo,
                examName:examName,
                orgUsername:orgUsername,
                question:req.body.question,
                solution:req.body.solution,
                maxMarks:req.body.maxMarks,
                coNo:coNo,
                level:level
                

            })
            question.save(function(err,doc){
                if(!err){
                    saveToExam(examName,orgUsername,req.body.questionNo);
                }
            });
            
        }
        
        
        res.redirect("/back");
    })
});

app.get("/back",function(req,res){
    if(req.user.role=="Admin"){
        res.redirect("/homeorganization");
    }
    else if(req.user.role=="Student"){
        res.redirect("/homestudent");
    }
    else if(req.user.role=="Teacher"){
        res.redirect("/hometeacher");
    }
})

/*



























*/



app.post("/giveExam",function(req,res){
    const examName=req.body.examName;
    const orgUsername=req.user.orgUsername;
    const n=Number(0);
    var date = new Date();
var currentDate = date.toISOString().slice(0,10);
var currentTime = date.getHours() + ':' + date.getMinutes();
Answer.findOne({examName:examName,orgUsername:orgUsername,studentUsername:req.user.username},function(err,foundAnswer){
    if(foundAnswer){
        res.render("unauthorized");
    }
    else{
        Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
            var d=foundExam.examDate;
            var eD=foundExam.examEndDate;
            var examD = d.toISOString().slice(0,10);
            var examT = d.getHours() + ':' + d.getMinutes();
            var examED = eD.toISOString().slice(0,10);
            var examET = eD.getHours() + ':' + eD.getMinutes();
            //if(currentDate>=examD && currentTime>=examT && currentDate<=examED && currentTime<=examET){
            if(1){
                if(foundExam){
                    
                    
        
                        Question.find({examName:examName,orgUsername:orgUsername},function(err,foundQuestion){
                            if(foundQuestion){

                                        res.render("giveExam",{Question:foundQuestion,Exam:foundExam,User:req.user,N:n,Ans:false});
                            }
                            else{
                                        res.render("giveExam",{Question:false,Exam:foundExam,User:req.user,N:0,Ans:false});
                            }
                        })
                }
            }
                
            else{
                res.render("unauthorized");
            }
        })
    }
})
});







app.post("/submitAnswer",function(req,res){
    //console.log(req.user);
    const studentUsername=req.user.username;
    const orgUsername=req.user.orgUsername;
    const examName=req.body.examName;
    const questionNo=req.body.questionNo;
    const answerNo=req.body.questionNo;
    //const question=req.body.question;
    const answer=req.body.answer;
    var n=Number(req.body.n);
    //console.log(studentUsername);
    //console.log(orgUsername);
    //console.log(examName);
    //console.log(questionNo);
    //console.log(answerNo);
    //console.log(question);
    //console.log(answer);
    
    var date = new Date();
    var currentDate = date.toISOString().slice(0,10);
    var currentTime = date.getHours() + ':' + date.getMinutes();
    Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
        if(foundExam){
    var d=foundExam.examDate;
    var eD=foundExam.examEndDate;
    var examD = d.toISOString().slice(0,10);
    var examT = d.getHours() + ':' + d.getMinutes();
    var examED = eD.toISOString().slice(0,10);
    var examET = eD.getHours() + ':' + eD.getMinutes();
    //if(currentDate>=examD && currentTime>=examT && currentDate<=examED && currentTime<=examET){
        if(1){
        Question.findOne({examName:examName,orgUsername:orgUsername,questionNo:questionNo},function(err,foundQuestion){
            Answer.findOne({examName:examName,orgUsername:orgUsername,studentUsername:req.user.username,questionNo:questionNo},function(err,foundAnswer){
                if(foundAnswer){
                    Question.find({examName:examName,orgUsername:orgUsername},function(err,foundQues){
                        if(foundQuestion){
                            console.log("Question Length= "+foundQues.length);
                            console.log("n value= "+n);
                            n=n+1;
                            console.log("n value= "+n);
                            console.log(foundQues[n]);
                            console.log(foundQues);
                            
                            if(foundQues.length>n){
                                Answer.find({examName:examName,orgUsername:orgUsername,studentUsername:req.user.username},function(err,foundAnsw){
                                    res.render("giveExam",{Question:foundQues,Exam:foundExam,User:req.user,N:n,Ans:foundAnsw});
                                })
                                
                                
                            }
                            else{
                                console.log("Successfully submitted all answers");
                                
                                res.redirect("/homestudent");
                            }
                            
                }
                else{
                            res.render("giveExam",{Question:false,Exam:foundExam,User:req.user,N:n});
                }
                    });
                }
                else{
                    const answ=new Answer({
                        studentUsername:studentUsername,
                        orgUsername:orgUsername,
                        examName:examName,
                        questionNo:questionNo,
                        answerNo:answerNo,
                        question:foundQuestion,
                        answer:answer
                    })
                    answ.save(function(err,doc){
                        if(!err){
                            saveAnsToExam(examName,orgUsername,questionNo);
                            Question.find({examName:examName,orgUsername:orgUsername},function(err,foundQues){
                                if(foundQuestion){
                                    console.log("Question Length= "+foundQues.length);
                                    console.log("n value= "+n);
                                    n=n+1;
                                    console.log("n value= "+n);
                                    console.log(foundQues[n]);
                                    console.log(foundQues);
                                    
                                    if(foundQues.length>n){
                                        Answer.find({examName:examName,orgUsername:orgUsername,studentUsername:req.user.username},function(err,foundAnsw){
                                        res.render("giveExam",{Question:foundQues,Exam:foundExam,User:req.user,N:n,Ans:foundAnsw});
                                        })
                                        
                                    }
                                    else{
                                        console.log("Successfully submitted all answers");
                                        
                                        res.redirect("/homestudent");
                                    }
                                    
                        }
                        else{
                                    res.render("giveExam",{Question:false,Exam:foundExam,User:req.user,N:n});
                        }
                            });
                            }
                            else{
                                console.log(err);
                            }
                               
                        })
                }
            });
            
                
        })
        
        

    }
}
})
});



app.get("/allResult",function(req,res){
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){
            const orgUsername=req.user.orgUsername;
            Answer.find({orgUsername:orgUsername},function(err,foundAns){
                if(!err){
                    
                    res.render("allSubmission",{Ans:foundAns,User:req.user});
                }
                else{
                    res.render("allSubmission",{Ans:false,User:req.user});
                }
            })
        }
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.render("unauthorized");
    }
});





app.post("/getaddCO",function(req,res){
    const examName=req.body.examName;
    //console.log("examName= "+examName);
    const orgUsername=req.user.orgUsername;
    //console.log("orgUsername= "+orgUsername);
    if(req.isAuthenticated()){
        if((req.user.role=="Admin"||req.user.role=="Teacher")&&(req.user.registeredOrg==true||req.user.registeredTeacher==true)){

            Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                if(foundExam){
                    CO.find({examName:examName,orgUsername:orgUsername},function(err,foundCo){
                        if(foundCo){
                            //console.log(foundExam);
                            res.render("addCo",{User:req.user,Exams:foundExam,Co:foundCo,message:""});
                        }
                        else{
                            res.render("addCo",{User:req.user,Exams:foundExam,Co:false,message:""});
                        }
                    })
                
                }
                else{
                    console.log("No exam found with this in app.get(/addQuestion)");
                    res.redirect("/back");
                }
                
            })
            
        }
        
        else{
            res.render("unauthorized");
        }
    }
    else{
        res.redirect("/");
    }


});







app.post("/addCo",function(req,res){
    const examName=req.body.examName;
    //console.log(examName);
    const orgUsername=req.user.orgUsername;
    CO.findOne({examName:examName,orgUsername:orgUsername,coNo:req.body.coNo},function(err,foundCo){
        if(foundCo){
            Exam.find({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                res.render("addCo",{User:req.user,Exams:foundExam,Co:foundCo,message:"CO Number Already Exist"});
            })
            
        }
        else{
            const co=new CO({
                coNo:req.body.coNo,
                examName:examName,
                orgUsername:orgUsername,
                co:req.body.co
            })
            co.save(function(err,doc){
                if(!err){
                    saveCoToExam(examName,orgUsername,req.body.coNo);
                }
            });
            
        }
        
        
        res.redirect("/back");
    })
})







app.post("/removeQuestion",function(req,res){
    const examName=req.body.examName;
    const orgUsername=req.user.orgUsername;
    const questionNo=req.body.questionNo;
    
    Question.find({examName:examName,orgUsername:orgUsername,questionNo:questionNo}).remove().exec();
    //Exam.updateMany({examName:examName,orgUsername:orgUsername},{$pull:{questions:{questionNo:questionNo,examName:examName,orgUsername:orgUsername}}});
    if(req.user.role=="Admin"){
        res.redirect("/homeorganization");
    }
    else{
        res.redirect("/hometeacher");
    }
})



app.post("/editQuestion",function(req,res){
    const examName=req.body.examName;
    const orgUsername=req.user.orgUsername;
    const questionNo=req.body.questionNo;
    Question.findOne({examName:examName,orgUsername:orgUsername,questionNo:questionNo},function(err,foundQue){
        if(!err){
            Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                if(!err){
                    CO.find({examName:examName,orgUsername:orgUsername},function(err,foundCo){
                        if(foundCo){
                            res.render("editQue",{question:foundQue,Exams:foundExam,Co:foundCo});
                        }
                        else{
                            res.render("editQue",{question:foundQue,Exams:foundExam,Co:false});
                        }
                    })
                }
                else{
                    console.log(err);
                }
            })

        }
        else{
            console.log(err);
        }
        
    })
})


app.post("/viewQP",function(req,res){
    const examName=req.body.examName;
    //console.log(examName);
    const orgUsername=req.user.orgUsername;
    
        
        
            Exam.findOne({examName:examName,orgUsername:orgUsername},function(err,foundExam){
                if(!err){
                    Question.find({examName:examName,orgUsername:orgUsername},function(err,foundQues){
                        if(!err && req.user.role!="Student"){
                            Organization.findOne({"orgUsername":orgUsername,role:"Admin",registeredOrg:true},function(err,foundOrg){
                                if(!err){
                                    res.render("viewQP",{User:req.user,Exams:foundExam,Question:foundQues,Org:foundOrg});
                                }
                                else{
                                    res.render("unauthorized");
                                }
                            })
                            
                        }
                        else{
                            res.render("unauthorized");
                        }
                    })
                }
                
            })
            
        
})







app.post("/editedQuestion",function(req,res){
    const examName=req.body.examName;
    const coNo=req.body.coNo;
    const level=predictLevel(req.body.question);
    var cos="";
    //console.log(examName);
    const orgUsername=req.user.orgUsername; 
    Question.findOneAndUpdate({examName:examName,orgUsername:orgUsername,questionNo:req.body.questionNo},{$set:{questionNo:req.body.questionNo,
        examName:examName,
        orgUsername:orgUsername,
        question:req.body.question,
        solution:req.body.solution,
        maxMarks:req.body.maxMarks,
        coNo:coNo,
        level:level}},{upsert:true},(error,doc)=>{
            console.log(error);
            if(!error){
                saveToExam(examName,orgUsername,req.body.questionNo);
                res.redirect("/back");
            }
            else{
                res.redirect("/back");
            }
        });

        
})

app.get("/autoEvaluate",function(req,res){
    exec('python autoEvaluate.py', (err, stdout, stderr) => {
        if (err) {
          console.error(err);
          res.redirect("/allResult");
          return;
        }
        else{
            console.log(stdout);
            res.redirect("/allResult");
        }
        
      });
})







app.listen(3000,function(){
    console.log("server started on port 3000");
});
/*const answerSchema=new mongoose.Schema({
    studentUsername:{type:String,sparse:true,unique:false},
    orgUsername:{type:String,sparse:true,unique:false},
    examName:{type:String,sparse:true,unique:false},
    questionNo:{type:String,sparse:true,unique:false},
    question:{type:questionSchema,sparse:true,unique:false},
    answerNo:{type:String,sparse:true,unique:false},
    answer:{type:String,sparse:true,unique:false},
})*/


