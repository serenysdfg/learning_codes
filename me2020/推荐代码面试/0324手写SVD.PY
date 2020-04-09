  #手写SVD
    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user]
                i = self.data.item[item]
                error = rating-self.predict(user,item)
                self.loss+=error**2
                p = self.P[u]
                q = self.Q[i]

                bu = self.Bu[u]
                bi = self.Bi[i]

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()\
               +self.regB*((self.Bu*self.Bu).sum()+(self.Bi*self.Bi).sum())
            iteration += 1
            self.isConverged(iteration)