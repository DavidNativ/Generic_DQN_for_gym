import numpy as np
import time



class Trainer:
    def __init__(self, name, env, agent, epsilon, eps_decay, eps_min, nb_epochs, log_interval, step_valid, nb_valid, render_valid, nb_test, render_test):
        self.env = env
        self.max_steps_per_game = 1000
        self.agent = agent

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.nb_epochs = nb_epochs
        self.step_valid = step_valid
        self.nb_valid = nb_valid
        self.render_val = render_valid
        self.log_consol_interval = round(self.nb_epochs / log_interval)

        self.nb_test = nb_test
        self.render_test = render_test

        self.name = name
        self.backup_interval = round(self.nb_epochs / 2)

        self.tab_running = []
        self.tab_avg_score = []
        self.tab_eval = []
        self.tab_eps = []
        self.tab_loss = []
        self.tab_lr = []

    def episode_training(self):
        observation = self.env.reset()
        score = 0.
        loss = 0.
        for n in range(self.max_steps_per_game):
            #action
            action = self.agent.act(observation, self.epsilon, training=True)
            #observation
            next_observation, reward, done, _ = self.env.step(action)
            # remember
            self.agent.remember(observation, action, reward, next_observation, done)
            #train
            l = self.agent.learn()
            #stats
            loss += l if l is not None else 0
            score += reward #stats
            #next step
            observation = next_observation
            if done:
                self.agent.scheduler.step()
                self.agent.update_target()
                break

        #batch training at the end of each episod
        #l = agent.long_term_training()
        #loss += l if l is not None else loss
        return score, loss/n

    def episode_testing(self, render):
        observation = self.env.reset()
        score = 0.
        for n in range(self.max_steps_per_game):
            action = self.agent.act(observation, 0, training=False)
            if render:
                self.env.render()
            next_observation, reward, done, _ = self.env.step(action)
            score += reward
            observation = next_observation
            if done:
                #print(f"--> score {score}")
                break
        return score


    def epochs(self):
        debut = time.time()
        begin_episode = time.time()

        current_tab_score = []
        current_tab_loss = []

        for e in range(1, self.nb_epochs+1):
            score, loss = self.episode_training()
            #LOG
            current_tab_score.append(score)
            current_tab_loss.append(loss)



            # console log, validation and saving steps
            if e > 0:
                if e % self.log_consol_interval == 0:
                    #on moyenne sur l'intervalle de log console
                    loss_ = np.mean(current_tab_loss[-self.log_consol_interval:])
                    score_ = round(np.mean(current_tab_score[-self.log_consol_interval:]), 0)
                    lr = self.agent.scheduler.get_last_lr()

                    print(f"Epoque {e:3d} : Avg Loss= {loss_} -- Score= {score_}"
                          f" -- eps= {round(self.epsilon, 4)} -- LR= {lr} -- ({round(time.time() - begin_episode, 2)}s.)")
                    begin_episode = time.time()
                    self.tab_loss.append(loss_)
                    self.tab_running.append(score_)
                    self.tab_eps.append(self.epsilon)
                    self.tab_lr.append(lr)


                #validation
                if e %  self.step_valid == 0:
                    s = 0
                    for _ in range(self.nb_valid):
                        score = self.episode_testing(self.render_val)
                        #print(f'--> {round(score, 0)}')
                        s += score
                    print(f"---- Eval : Avg Eval Score= {round(s / self.nb_valid, 0)} -- Training avg score (/100 epochs)={round(np.mean(self.tab_running[-100:]), 0)}")
                    print()
                    self.tab_eps.append(self.epsilon)
                    self.tab_eval.append(s / self.nb_valid)
                    self.tab_avg_score.append(np.mean(self.tab_running[-100:]))
                    self.env.close()


                #backup model
                if e % self.backup_interval == 0:
                    y, m, d, h, mi, s, _, _, _ = time.localtime()
                    filename = f"./MODELS/{self.name}_{y}-{m}-{d}_{h}-{mi}-{s}_{e}-{self.nb_epochs}"
                    self.agent.action_value.save(filename)


            # epsilon decay
            self.epsilon -= self.eps_decay if self.epsilon >= self.eps_min else 0

        #end of the training epochs
        self.env.close()
        tps_ecoule = round(time.time() - debut,2)
        print(f"Temps Total ecoule pour {self.nb_epochs} parties: {tps_ecoule}s.")

        y, m, d, h, mi, s, _, _, _ = time.localtime()
        filename = f"./MODELS/{self.name}_{y}-{m}-{d}_{h}-{mi}-{s}_FINAL-{self.nb_epochs}"
        self.agent.action_value.save(filename)
        print(f"\nSauvegarde {filename} ... \n")

        # final tests
        s = 0
        correct = 0
        for i in range(self.nb_test):
            score = self.episode_testing(self.render_test)
            #print(f"----> Test Score {i+1}/{self.nb_test} : {round(score)}")
            s += score
            if score >= 200:
                correct += 1
        final_score = round(s / self.nb_test, 0)
        print(f"\n---- FINAL TEST SCORE : {final_score} -- Victories: {correct}")

        self.env.close()
        return final_score, tps_ecoule, correct

    def get_stats(self):
        return self.tab_running, self.tab_avg_score, self.tab_eval, self.tab_eps, self.tab_loss, self.tab_lr
