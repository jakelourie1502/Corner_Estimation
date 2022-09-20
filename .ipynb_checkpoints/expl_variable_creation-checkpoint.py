import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import pandas as pd

### LEAGUE AND TEAM OBJECTS
class League:
    def __init__(self):
        self.home_mean = 0
        self.away_mean = 0
        self.teams_list = []
        self.teams = {} #to store team objects next to their ID
        self.table = False

class Team:
    def __init__(self):
        #sporting performance index(spi) for 4 relevant values.
        self.H_attacking_corners_spi = 0
        self.H_defending_corners_spi = 0
        self.A_attacking_corners_spi = 0
        self.A_defending_corners_spi = 0
        self.H_attacking_goals_spi = 0
        self.H_defending_goals_spi = 0
        self.A_attacking_goals_spi = 0
        self.A_defending_goals_spi = 0
        self.home_count = 0 
        self.away_count = 0
        self.total_count = 0

#### FUNCTION 1 TO CREATE THE LEAGUE AND TEAMS OBJECTS
def initialisate_leagues(uniques_leagues, only_train_df,zero_means=False):
    """
    This function takes all the different league IDs and a train or validation dataframe.
    It returns:
        a dictionary ("leagues") with the relevant tables, an initialisation mean values for home and away corners.
        list of teams and dictionary of team id : team objects.
        NOTE: zero_means prevents there being an update to the means leagues initialisation. This is for when we we are using the poisson distribution and require only positive values for explanatory variables.
    """
    leagues = {}
    idx = 0.
    for ul in uniques_leagues:
        leagues[ul] = League() #create league object
        leagues[ul].idx = idx
        idx+=1
        leagues[ul].table = only_train_df[only_train_df['LeagueId']==ul].reset_index().drop('index',axis=1) #filter table
        # leagues[ul].home_mean = np.mean(leagues[ul].table['Home_Corners']) #create mean for home corners
        # leagues[ul].away_mean = np.mean(leagues[ul].table['Away_Corners']) #create mean for away corners
    
        leagues[ul].teams_list = list(set(list(leagues[ul].table['HomeTeamId'])+list(leagues[ul].table['AwayTeamId'])))  #create list of teams in the list
        leagues[ul].teams = {}
        #create a dictioanry which has team id and a team object in it.
        for team in leagues[ul].teams_list:
            leagues[ul].teams[team] = Team()
    leagues['average_cor_home'] = np.mean(only_train_df['Home_Corners']) if not zero_means else 0
    leagues['average_cor_away'] = np.mean(only_train_df['Away_Corners']) if not zero_means else 0
    leagues['average_goal_home'] = np.mean(only_train_df['Home_Goals']) if not zero_means else 0
    leagues['average_goal_away'] = np.mean(only_train_df['Away_Goals']) if not zero_means else 0
    
    return leagues


#### FUNCTION 2 - TO CREATE DATASETS EVERY TIME A MATCH IS PLAYED THE TEAM STATS MUST BE UPDATED
def perform_update(leagues, match, league, rho=0.95, rho_league=0.999,unit_test=False,zero_means=True):
    """
    Takes in a match and a league object.
    Store_early_vals determines whether to update the 'early value' for a specific team.
    updates:
        home and & away team SPIs.
        league averages.
        
    NOTE: zero_means prevents there being an update to the means leagues initialisation. This is for when we we are using the poisson distribution and require only positive values for explanatory variables.
    """
    #Extract values for easy from dictionary and team/league objects
    def extract_home_away_teams(match, leagues):
        home_team_id = match[1]['HomeTeamId']
        home_team = league.teams[home_team_id]
        away_team_id = match[1]['AwayTeamId']
        away_team = league.teams[away_team_id]
        return home_team, away_team
    
    def extract_league_averages(leagues):
    
        home_cor_league_avg = leagues['average_cor_home']
        away_cor_league_avg = leagues['average_cor_away']
        home_goal_league_avg = leagues['average_goal_home'] 
        away_goal_league_avg = leagues['average_goal_away']
        return home_cor_league_avg, away_cor_league_avg, home_goal_league_avg, away_goal_league_avg
    
    def unit_test1():
        print("\n\n====UNIT TEST====")
        print(f"league home mean: {home_cor_league_avg}") 
        print(f"league away mean: {away_cor_league_avg}") 
        print(f"\nhome team stats:")
        for k, v in vars(league.teams[home_team_id]).items():
            print(k,v)
        print(f"\naway team stats:")
        for k, v in vars(league.teams[away_team_id]).items():
            print(k,v)
        print(f"home_corners: {home_corners}")
        print(f"away_corners: {away_corners}")
        
    def unit_test2():
        print(f"New league home mean: {home_cor_league_avg}") 
        print(f"New league away mean: {away_cor_league_avg}") 
        print(f"\nnew home team stats:")
        for k, v in vars(league.teams[home_team_id]).items():
            print(k,v)
        print(f"\nnew away team stats:")
        for k, v in vars(league.teams[away_team_id]).items():
            print(k,v)
            

    #get home and away team obj
    home_team, away_team, = extract_home_away_teams(match, leagues)
    
    #get league averages
    home_cor_league_avg, away_cor_league_avg, home_goal_league_avg, away_goal_league_avg = extract_league_averages(leagues)
        
    #extract corners from this match
    home_corners,away_corners =match[1]['Home_Corners'], match[1]['Away_Corners']     
    home_goals, away_goals = match[1]['Home_Goals'], match[1]['Away_Goals']
    if math.isnan(home_goals) or math.isnan(away_goals):
        return False
    ## unit test: state of play before update
    if unit_test:        unit_test1()

    #Create "gradient" - a "good" result is always characterised by a positive d value.
    dHome_corner_attacking = home_corners - (home_cor_league_avg-away_team.A_defending_corners_spi) 
    dHome_corner_defending = -(away_corners - (away_cor_league_avg+away_team.A_attacking_corners_spi))
    dAway_corner_attacking = away_corners - (away_cor_league_avg-home_team.H_defending_corners_spi)
    dAway_corner_defending = -(home_corners - (home_cor_league_avg+home_team.H_attacking_corners_spi))
    
    dHome_goal_attacking = home_goals - (home_goal_league_avg-away_team.A_defending_goals_spi) 
    dHome_goal_defending = -(away_goals - (away_goal_league_avg+away_team.A_attacking_goals_spi))
    dAway_goal_attacking = away_goals - (away_goal_league_avg-home_team.H_defending_goals_spi)
    dAway_goal_defending = -(home_goals - (home_goal_league_avg+home_team.H_attacking_goals_spi))
    

    #Perform updates - team and league
    home_team.H_attacking_corners_spi = rho*home_team.H_attacking_corners_spi + (1-rho)*dHome_corner_attacking
    home_team.H_defending_corners_spi = rho*home_team.H_defending_corners_spi + (1-rho)*dHome_corner_defending
    away_team.A_attacking_corners_spi = rho*away_team.A_attacking_corners_spi + (1-rho)*dAway_corner_attacking
    away_team.A_defending_corners_spi = rho*away_team.A_defending_corners_spi + (1-rho)*dAway_corner_defending
    
    home_team.H_attacking_goals_spi = rho*home_team.H_attacking_goals_spi + (1-rho)*dHome_goal_attacking
    home_team.H_defending_goals_spi = rho*home_team.H_defending_goals_spi + (1-rho)*dHome_goal_defending
    away_team.A_attacking_goals_spi = rho*away_team.A_attacking_goals_spi + (1-rho)*dAway_goal_attacking
    away_team.A_defending_goals_spi = rho*away_team.A_defending_goals_spi + (1-rho)*dAway_goal_defending
    
    # print(home_goals,dHome_goal_attacking, home_team.H_attacking_goals_spi)
    ## update league averages
    if not zero_means:
        leagues['average_cor_home'] = rho_league * leagues['average_cor_home'] + (1-rho_league)*home_corners
        leagues['average_cor_away'] = rho_league * leagues['average_cor_away'] + (1-rho_league)*away_corners
        leagues['average_goal_home'] = rho_league * leagues['average_goal_home'] + (1-rho_league)*home_goals
        leagues['average_goal_away'] = rho_league * leagues['average_goal_away'] + (1-rho_league)*away_goals

    #update for bootstrapping
    home_team.home_count += 1; home_team.total_count += 1
    away_team.away_count += 1; away_team.total_count += 1
    
    ## unit test: state of play after update
    if unit_test:        unit_test2()
        
        


def get_x_values_one_team(team):
    """
    HELPER FUNCTION
    Takes a team, boostrap point (5 / 10) and mode
    returns the home and away stats for that team (attacking and defending)
    """
    
    h_att_cor, h_def_cor = team.H_attacking_corners_spi, team.H_defending_corners_spi
    A_att_cor, A_def_cor = team.A_attacking_corners_spi, team.A_defending_corners_spi
    h_att_goals, h_def_goals = team.H_attacking_goals_spi, team.H_defending_goals_spi
    A_att_goals, A_def_goals = team.A_attacking_goals_spi, team.A_defending_goals_spi
    return h_att_cor, h_def_cor ,  h_att_goals, h_def_goals, A_att_cor, A_def_cor,A_att_goals, A_def_goals


def create_training_set(uniques_leagues, only_train_df,bootstrap_value = 10, rho =0.95, rho_league=0.99, unit_test=False,zero_means=False):
    """
    Initialises all leagues
    creates SPIs by going through the train_df
    then does a second pass where it creates the dataset. Early matches uses a bootstrapped early SPI value
    Returns
        x
        y
        df check
    
    Unit_test argument: prints the matches for a given team
    """
    def unit_test1():
        print('\n===NEW_LINE===')
        print('home/away: above threshold - ' , home_team.home_count > bootstrap_value, away_team.away_count > bootstrap_value)
        print('\nhome team early values: ',home_team.early_values['home'])
        print('\nHome team SPIs ', home_team.H_attacking_corners_spi, home_team.H_defending_corners_spi)
        print('\nfinal values used for inputs: ',A_att, A_def)
    
    
    leagues = initialisate_leagues(uniques_leagues, only_train_df, zero_means=zero_means)
    
    x, y = [], []

    for key, l in leagues.items():
        if isinstance(l, League):
            l.table['H_ATT'], l.table['H_DEF'],l.table['H_ATT_goals'], l.table['H_DEF_goals'],\
                l.table['A_ATT'], l.table['A_DEF'], l.table['A_ATT_goals'], l.table['A_DEF_goals'] = 0,0,0,0,0,0,0,0
            h_att_idx = l.table.columns.get_loc('H_ATT')
            
            for row, match in enumerate(l.table.iterrows()):

                home_team, away_team = l.teams[match[1]['HomeTeamId']], l.teams[match[1]['AwayTeamId']]
                home_stats = get_x_values_one_team(home_team)
                away_stats = get_x_values_one_team(away_team)

                if match[1]['HomeTeamId'] == 1056 and unit_test:
                    print('hi')
                    unit_test1()                    
                
                perform_update(leagues, match, l, rho=rho, rho_league=rho_league, zero_means=zero_means)
                    
                if home_team.total_count > bootstrap_value and away_team.total_count > bootstrap_value:
                    x.append([*home_stats, *away_stats, l.idx]) #, l.home_mean, l.away_mean])
                    y.append([match[1]['Home_Corners'],match[1]['Away_Corners'],match[1]['total_corners']])

                    ##### appending dataset
                
                for r, stat in zip(range(8), [*home_stats[:4],*away_stats[:4]]):
                    l.table.iloc[row, h_att_idx+r] = np.abs(stat) if zero_means else stat
            
    
    if zero_means: #SPECIFICALLY FOR POISSON - TURNS -VES INTO POSITIVES 
        x = [np.abs(b) for b in [a for a in x]]
    return leagues, x, y 


def create_val_set(unique_leagues, val_df, leagues,zero_means):
    """
    Take the val_df and leagues object
    the leagues object has teams objects in it, with the SPI of the teams
    """
    val_leagues = initialisate_leagues(unique_leagues, val_df,zero_means=zero_means)
    x, y = [], []
    for key, l in val_leagues.items():
        if isinstance(l, League):
            #### amend the table for checking.
            l.table['H_ATT'], l.table['H_DEF'],l.table['H_ATT_goals'], l.table['H_DEF_goals'],\
                l.table['A_ATT'], l.table['A_DEF'], l.table['A_ATT_goals'], l.table['A_DEF_goals'] = 0,0,0,0,0,0,0,0
            h_att_idx = l.table.columns.get_loc('H_ATT')
            
            for row, match in enumerate(l.table.iterrows()):
                ### Extract teams      
                league_id = match[1]['LeagueId']       
                home_team_id = match[1]['HomeTeamId']
                away_team_id = match[1]['AwayTeamId']
                home_team, away_team = leagues[league_id].teams[home_team_id], leagues[league_id].teams[away_team_id]

                #extract values from the leagues object (from training)
                home_stats = get_x_values_one_team(home_team)
                away_stats = get_x_values_one_team(away_team)
                x.append([*home_stats, *away_stats,l.idx])#, leagues[league].home_mean, leagues[league].away_mean])
                y.append([match[1]['Home_Corners'],match[1]['Away_Corners'],match[1]['total_corners']])
                
                for r, stat in zip(range(8), [*home_stats[:4],*away_stats[:4]]):
                    l.table.iloc[row, h_att_idx+r] = np.abs(stat) if zero_means else stat
                     
    if zero_means: #SPECIFICALLY FOR POISSON - TURNS -VES INTO POSITIVES 
        x = [np.abs(b) for b in [a for a in x]]
    return  val_leagues,x, y