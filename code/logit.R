library(logistf)
library(psych)
# library("RMySQL")
# 
# con <- dbConnect(MySQL(),
#     user="", password="",
#     dbname="", host="")
# on.exit(dbDisconnect(con))
# 
# feature_set <- dbGetQuery(con, "SELECT * FROM feature_set")
feature_set <- read.csv('cscw_features.csv')

# PREDICT MERGES USING PULL REQUEST DATA
pr_logit.1 <- glm(merged ~ pr_comments_user, data=feature_set, family="binomial")
summary(pr_logit.1)
# add controls
pr_logit.1a <- update(pr_logit.1, .~. + repo_commits + repo_contributors)
# check for improvement - isn't any, so removed controls from here
anova(pr_logit.1,pr_logit.1a, test="Chisq")

pr_logit.2 <- update(pr_logit.1, .~. + reputation)
summary(pr_logit.2)

# including changes_on_pr produces separation, so I'm leaving it out for now
# pr_logit.3 <- update(pr_logit.2, .~. + changes_on_pr)
# summary(pr_logit.3)
pr_logit.4 <- update(pr_logit.2, .~. + pr_comments_other)
summary(pr_logit.4)

# see if our model actually improved
anova(pr_logit.2,pr_logit.4, test="Chisq") 

# get odds-ratios for predictors in best model
exp(cbind(OR = coef(pr_logit.4), confint(pr_logit.4)))

# PREDICT MERGES USING ISSUE DATA
i_logit.1 <- glm(merged ~ issue_comments, data=feature_set, family="binomial")
summary(i_logit.1)
i_logit.2 <- update(i_logit.1, .~. + issues_opened)
summary(i_logit.2)
i_logit.3 <- update(i_logit.2, .~. + reputation)
summary(i_logit.3)

# see if our model actually improved
anova(i_logit.1,i_logit.2, test="Chisq") 
anova(i_logit.2,i_logit.3, test="Chisq") 

# get odds-ratios for predictors in best model
exp(cbind(OR = coef(i_logit.2), confint(i_logit.2)))

# PREDICT MERGES USING ALL THE DATA WE HAVE
# start with dev comments on both PR and I
big_logit.1 <- glm(merged ~ pr_comments_user + issue_comments + issues_opened + reputation, data=feature_set, family="binomial")
summary(big_logit.1)

# add community response
big_logit.2 <- update(big_logit.1, .~. + pr_comments_other)
summary(big_logit.2)

# check for improvement
anova(big_logit.1, big_logit.2, test="Chisq")

# add controls
big_logit.3 <- update(big_logit.2, .~. + repo_commits + repo_contributors)
summary(big_logit.3)

# check for improvement - controls don't improve this model either
anova(big_logit.2, big_logit.3, test="Chisq")

# get odds-ratios for predictors in best model
exp(cbind(OR = coef(big_logit.2), confint(big_logit.2)))

# check for improvement over PR and I models
anova(big_logit.2, pr_logit.4, test="Chisq")
anova(big_logit.2, i_logit.2, test="Chisq")

# make our pretty table
or.vector1 <- exp(pr_logit.1$coef)
or.vector2 <- exp(pr_logit.2$coef)
or.vector3 <- exp(pr_logit.4$coef)
or.vector4 <- exp(i_logit.2$coef)
or.vector5 <- exp(i_logit.3$coef)
or.vector6 <- exp(big_logit.2$coef)
ci.vector1 <- exp(confint(pr_logit.1))
ci.vector2 <- exp(confint(pr_logit.2))
ci.vector3 <- exp(confint(pr_logit.4))
ci.vector4 <- exp(confint(i_logit.2))
ci.vector5 <- exp(confint(i_logit.2))
ci.vector6 <- exp(confint(big_logit.2))
stargazer(pr_logit.1,pr_logit.2,pr_logit.4,i_logit.2,i_logit.3,big_logit.2,
          coef=list(or.vector1,or.vector2,
                    or.vector3,or.vector4,
                    or.vector5,or.vector6),
          ci = T,
          ci.custom = list(ci.vector1,ci.vector2,
                           ci.vector3,ci.vector4,
                           ci.vector5,ci.vector6),
          dep.var.caption = "", 
          dep.var.labels.include = F,
          digits = 2,
          title="Logisitic regressions: Predicting pull request merges (odds ratio and confidence intervals)")
stargazer(pr_logit.1,pr_logit.2,pr_logit.4,i_logit.2,i_logit.3,big_logit.2, 
          dep.var.caption = "", 
          dep.var.labels.include = F,
          title="Logisitic regressions: Predicting pull request merges (coefficients and standard error)")
