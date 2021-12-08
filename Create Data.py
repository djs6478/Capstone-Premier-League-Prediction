#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import glob
import numpy as np
import pandas as pd


# In[1]:


class ParseRawData:
    def __init__(self, filename):
        with open(filename) as file:
            self.content = file.readlines()
            
    def _is_date(self, line):
        day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        return any(map(lambda d: d in line, day))
    
    def _parse_date(self, line_date):
        """
        e.g. line_date = 'Sunday 22 May 2011\n' 
        return : ('2011-05-22', 'Sunday')
        """
        weekday = line_date.split()[0]
        date = pd.to_datetime(line_date).strftime('%Y-%m-%d')
        return date, weekday
    
    def _parse_match(self, i):
        """
        e.g.
        ====
        Chelsea     <---- i should be here
        2-0 
        Wolves
         Stamford Bridge, 
        London
                    <---- returned i is here
        Crystal Palace
        1-1 
        Spurs
         Selhurst Park, 
        London
        """
        content = self.content
        match_info = []    # List: [HomeTeam, Result, AwayTeam, Stadium, City]
        
        while len(match_info) < 5 and i < len(content):
            line = content[i]
            if line.strip():
                match_info.append(line.strip())
            i += 1
                
        home_team = match_info[0]
        away_team = match_info[2]
        home_score, away_score = match_info[1].split('-')
        stadium = match_info[3]
        city = match_info[4]
        
        if home_score > away_score:
            winner = home_team
        elif home_score < away_score:
            winner = away_team
        elif home_score == away_score:
            winner = 'Draw'
        
        # returned match info
        rmatch_info = [home_team, int(home_score), int(away_score), away_team, winner, stadium.strip(','), city]
        
        return i, rmatch_info
        
    def parse(self):
        content = self.content
        length = len(content)
        
        matchs_info = []    # List_matchs[List[str]]
        i = 0
        while i < length:
            line = content[i]
            
            if line.strip():
                
                if self._is_date(line):
                    date, weekday = self._parse_date(line)
                    i += 1

                else:
                    i, match_info = self._parse_match(i)
                    matchs_info.append([date, weekday] + match_info)
                    
            else:
                i += 1
                
        df = pd.DataFrame(matchs_info)
        df.columns = ['Date', 'Weekday', 'HomeTeam', 'HomeScore', 'AwayScore', 'AwayTeam', 'Winner', 'Stadium', 'City']
        return df
    
    
if __name__ == '__main__':
    # test the parse result
    test = ParseRawData('./raw_data/1920.txt').parse()
    print('Test parsing 1920 season raw data')
    print('---------------------------------')
    print(test)
    print()
    
    print('Start to parse all data')
    print('----------------')
    allseason = sorted(glob.glob('raw_data/*.txt'))

    for season in allseason:
        print(season)

        # e.g. \\raw_data\\1011.txt -> 1011.csv
        savefilename = season.split('\\')[1].replace('txt', 'csv')

        df = ParseRawData(season).parse()
        df.to_csv(f'./clean_data/{savefilename}')


# In[2]:


class MakeTeamData:
    def __init__(self, filename):
        """ e.g. filename = './clean_data/1819.csv' """
        self.df_season = pd.read_csv(filename, index_col=0)
        self.allteams = sorted(self.df_season['HomeTeam'].unique())
        
        # e.g. filename = './clean_data/1819.csv'  -->  foldername = '1819'
        self.foldername = re.findall('.+/([0-9]{4}).csv', filename)[0]
        
        try:
            os.mkdir(f'./team_data/{self.foldername}')
        except FileExistsError:
            pass
        
    def _parse_team(self, team):
        """team: str, return: df of this team"""
        df_team = self.df_season[(self.df_season['HomeTeam'] == team) | (self.df_season['AwayTeam'] == team)].copy()
        df_team['Date'] = pd.to_datetime(df_team['Date'])
        df_team = df_team.sort_values(by='Date')
        df_team = df_team.reset_index(drop=True)
        
        # extract information from df_team to df2_team
        df2_team = df_team[['Date']].copy()

        # round
        df2_team['Round'] = list(range(1, df_team.shape[0]+1, 1))

        # if is home match
        ishome = df_team['HomeTeam'] == team
        df2_team['isHome'] = ishome

        # the rival team
        df2_team['Rival'] = pd.concat((
            df_team.loc[ishome, 'AwayTeam'], 
            df_team.loc[~ishome, 'HomeTeam']
        )).sort_index()
        
        # goals in the match
        df2_team['Goal'] = pd.concat((
            df_team.loc[ishome, 'HomeScore'],
            df_team.loc[~ishome, 'AwayScore']
        )).sort_index()

        # conceded in the match
        df2_team['Conceded'] = pd.concat((
            df_team.loc[ishome, 'AwayScore'],
            df_team.loc[~ishome, 'HomeScore']
        )).sort_index()

        # points earned in the match
        df2_team['Points'] = df_team['Winner']
        df2_team['Points'].replace(to_replace=team, value=3, inplace=True)
        df2_team['Points'].replace(to_replace='Draw', value=1, inplace=True)
        df2_team.loc[df2_team.Points.str.isnumeric() == False, 'Points'] = 0

        # cumulative points including this match
        df2_team['CumPoints'] = df2_team['Points'].cumsum()
        
        # cumulative points excluding this match
        df2_team['bCumPoints'] = df2_team['CumPoints'] - df2_team['Points']
        
        # points earned in the last five matches (excluding this one)
        df2_team['b5MatchPoints'] = df2_team['Points'].rolling(6, min_periods=1).sum() - df2_team['Points']
        
        # normalized points (=1: earned all points in the past five matches) earned in the last five matches (excluding this one)
        b5_max_points = df2_team['Round'].rolling(6, min_periods=1).count() * 3 - 3
        b5_max_points[b5_max_points == 0] = np.nan
        df2_team['b5MatchPointRatio'] = df2_team['b5MatchPoints'] / b5_max_points
        
        # cumulative goals (excluding this matches)
        df2_team['bCumGoal'] = df2_team['Goal'].cumsum() - df2_team['Goal']
        
        # cumulative goals in the past five matches (excluding this one)
        df2_team['b5MatchGoal'] = df2_team['Goal'].rolling(6, min_periods=1).sum() - df2_team['Goal']
        
        # cumulative conceded (excluding this matches)
        df2_team['bCumConceded'] = df2_team['Conceded'].cumsum() - df2_team['Conceded']
        
        # cumulative conceded in the past five matches (excluding this one)
        df2_team['b5MatchConceded'] = df2_team['Conceded'].rolling(6, min_periods=1).sum() - df2_team['Conceded']
        
        ### only consider the matches at home
        # points earned in the last five home matches (excluding this one)
        df2_team_home = df2_team[df2_team.isHome].copy()
        df2_team_home['b5HomeMatchPoints'] = df2_team_home['Points'].rolling(6, min_periods=1).sum() - df2_team_home['Points']
        
        # normalized points earned in the last five home matches (excluding this one)
        b5_max_points = df2_team_home['Round'].rolling(6, min_periods=1).count() * 3 - 3
        b5_max_points[b5_max_points == 0] = np.nan
        df2_team_home['b5HomeMatchPointRatio'] = df2_team_home['b5HomeMatchPoints'] / b5_max_points
        
        # cumulative goals in the home matches (excluding this match)
        df2_team_home['bHomeCumGoal'] = df2_team_home['Goal'].cumsum() - df2_team_home['Goal']
        
        # cumulative conceded in the home matches (excluding this match)
        df2_team_home['bHomeCumConceded'] = df2_team_home['Conceded'].cumsum() - df2_team_home['Conceded']
        
        # cumulative goals in the last five home matches (excluding this match)
        df2_team_home['b5HomeMatchGoal'] = df2_team_home['Goal'].rolling(6, min_periods=1).sum() - df2_team_home['Goal']
        
        # cumulative conceded in the last five home matches (excluding this match)
        df2_team_home['b5HomeMatchConceded'] = df2_team_home['Conceded'].rolling(6, min_periods=1).sum() - df2_team_home['Conceded']
        
        ### only consider the matches at away
        # points earned in the last five away matches (excluding this one)
        df2_team_away = df2_team[~df2_team.isHome].copy()
        df2_team_away['b5AwayMatchPoints'] = df2_team_away['Points'].rolling(6, min_periods=1).sum() - df2_team_away['Points']
        
        # normalized points earned in the last five away matches (excluding this one)
        b5_max_points = df2_team_away['Round'].rolling(5, min_periods=1).count() * 3 - 3
        b5_max_points[b5_max_points == 0] = np.nan
        df2_team_away['b5AwayMatchPointRatio'] = df2_team_away['b5AwayMatchPoints'] / b5_max_points
        
        # cumulative goals in the away matches (excluding this match)
        df2_team_away['bAwayCumGoal'] = df2_team_away['Goal'].cumsum() - df2_team_away['Goal']
        
        # cumulative conceded in the away matches (excluding this match)
        df2_team_away['bAwayCumConceded'] = df2_team_away['Conceded'].cumsum() - df2_team_away['Conceded']
        
        # cumulative goals in the last five away matches (excluding this match)
        df2_team_away['b5AwayMatchGoal'] = df2_team_away['Goal'].rolling(6, min_periods=1).sum() - df2_team_away['Goal']
        
        # cumulative conceded in the last five away matches (excluding this match)
        df2_team_away['b5AwayMatchConceded'] = df2_team_away['Conceded'].rolling(6, min_periods=1).sum() - df2_team_away['Conceded']
        
        ### concat the dataframes
        df2_team = pd.concat(
            [
                df2_team,
                df2_team_home.drop(labels=df2_team.columns, axis=1),
                df2_team_away.drop(labels=df2_team.columns, axis=1)
            ],
            axis=1
        )
        
        return df2_team
        
    def parse(self):
        for team in self.allteams:    
            print(team + ' ...', end=' ')
            df_team = self._parse_team(team)
            df_team.to_csv(f'./team_data/{self.foldername}/{team}.csv', index=False)
            print('[Done]')
            
            
if __name__ == '__main__':
    # test the parse result
    df = MakeTeamData('clean_data/1819.csv')._parse_team('Liverpool')
    print('Test parsing Liverpool data at 1819 season')
    print('------------------------------------------')
    print(df.iloc[:10,:15])
    print()
    
    print('Start to parse all data')
    print('----------------------')
    csvfiles = glob.glob('./clean_data/*.csv')
    csvfiles = list(map(lambda cfile: cfile.replace('\\', '/'), csvfiles))

    for cfile in csvfiles:
        print(f' =========== {cfile} ============')
        MakeTeamData(cfile).parse()
        print()


# In[3]:


def extract_team_df_info(df):
    win = (df['Points'] == 3).sum()
    draw = (df['Points'] == 1).sum()
    loss = (df['Points'] == 0).sum()
    goals = f'{df["Goal"].sum()}:{df["Conceded"].sum()}'
    points = df['Points'].sum()
    
    idx_home = df['isHome']
    win_home = (df.loc[idx_home, 'Points'] == 3).sum()
    draw_home = (df.loc[idx_home, 'Points'] == 1).sum()
    loss_home = (df.loc[idx_home, 'Points'] == 0).sum()
    goals_home = f'{df.loc[idx_home, "Goal"].sum()}:{df.loc[idx_home, "Conceded"].sum()}'
    
    win_away = (df.loc[~idx_home, 'Points'] == 3).sum()
    draw_away = (df.loc[~idx_home, 'Points'] == 1).sum()
    loss_away = (df.loc[~idx_home, 'Points'] == 0).sum()
    goals_away = f'{df.loc[~idx_home, "Goal"].sum()}:{df.loc[~idx_home, "Conceded"].sum()}'
    
    return_tup = (
        win, draw, loss, goals, points,
        win_home, draw_home, loss_home, goals_home,
        win_away, draw_away, loss_away, goals_away
    )
    return return_tup


def create_table(season):
    print(f'===== Read csv files at team_data/{season}/*.csv =====')
    teams_csv = glob.glob(f'team_data/{season}/*.csv')
    
    table = []    # List[Tuple(team_name, points, home_goals, home_conceded, away_goals, away_conceded)]
    
    for team_csv in teams_csv:        
        df = pd.read_csv(team_csv)     
        info = extract_team_df_info(df)
        #points = df.iloc[-1, df.columns.get_loc('CumPoints')]
        team_name = re.findall('([A-Za-z ]+).csv$', team_csv)[0]
        table.append((team_name, *info))
        
    table = pd.DataFrame(
        table, 
        columns=[
            'Team', 'Win', 'Draw', 'Loss', 'Goals', 'Points',
            'WinHome', 'DrawHome', 'LossHome', 'GoalsHome',
            'WinAway', 'DrawAway', 'LossAway', 'GoalsAway'
        ]
    )
    table = table.sort_values(by='Points', ascending=False, ignore_index=True)
    table.insert(0, 'Rank', table.index+1)
    return table


if __name__ == '__main__':
    allseason = list(filter(lambda f: f.isdigit(), os.listdir('team_data')))
    
    for season in allseason:
        table = create_table(season)
        table.to_csv(f'table/{season}.csv', index=False)
        print(f' Save table at table/{season}.csv')
        print()


# In[4]:


def create_table(season):
    dfs_dict = dict()
    
    for table_type in ['all', 'home', 'away']:
        path = f'table/Championship/raw_txt/{table_type}/{season}'
        with open(path) as file:
            content = file.readlines()
            
        df = pd.DataFrame(
            np.array(content).reshape(-1, 9),
            columns=['Rank', 'FullName', 'Name', 'PlayedMatchs', 'Win', 'Draw', 'Loss', 'Goals', 'Points']
        )
        df = df.applymap(lambda s: s.strip())
        intcol = ['PlayedMatchs', 'Win', 'Draw', 'Loss', 'Points']
        df[intcol] = df[intcol].applymap(lambda s: int(s))
        
        dfs_dict[table_type] = df
        
    df_final = dfs_dict['all'].copy()
    df_home = dfs_dict['home'].copy()
    df_away = dfs_dict['away'].copy()
    
    df_final = df_final.merge(
        df_home[['FullName', 'Win', 'Draw', 'Loss', 'Goals']], 
        on='FullName', 
        suffixes=('', 'Home')
    ).merge(
        df_away[['FullName', 'Win', 'Draw', 'Loss', 'Goals']], 
        on='FullName',
        suffixes=('', 'Away')
    )
    
    df_final['Name'] = df_final['Name'].str.replace('Wolverhampton', 'Wolves')
    return df_final
    
    
if __name__ == '__main__':
    seasons = list(filter(lambda s: s.endswith('.txt'), os.listdir('table/Championship/raw_txt/all/')))

    for season in seasons:
        print(f'season {season} -- ', end='')
        table = create_table(season)
        table.to_csv(f'table/Championship/csv/{season.replace("txt", "csv")}', index=False)
        print('[done]')


# In[5]:


# calculate attack / defense strength based on the previous season
# reference: https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/MD62MLXUMKMXZ6A8
class Strength:
    """
    Compute attck / defence strength
    """
    
    def __init__(self, season):
        """
        e.g. season=1920
        And it will calculate the attack/defence strength based on the statistics
        of the last (1819) season.
        level = 0 for Premier League, 1 for Championship League
        """
        self.season = season
        
    def _get_all_teams(self, season):
        allteams_path = glob.glob(f'team_data/{season}/*.csv')
        allteams = list(map(lambda s: re.findall('([A-Za-z ]+).csv$', s)[0], allteams_path))
        return allteams
        
    def _read_file(self, level):
        """return df (table of the last season)"""
        y1 = int(self.season[:2])
        y2 = int(self.season[2:])
        last_season = f'{y1-1:02d}{y2-1:02d}'
        
        if level == 0:
            return pd.read_csv(f'table/{last_season}.csv')
        elif level == 1:
            return pd.read_csv(f'table/Championship/csv/{last_season}.csv')
        else:
            raise ValueError('self.level should be 0 (Premier league) or 1 (Championship league)')
        
    def calc_strength(self, level):
        """
        The result is based on the previous season.
        ASH : Attack Strength at Home
        ASA : Attack Strength Away
        DSH : Defence Strength at Home
        DSA : Defence Strength Away
        """
        # table of last season
        df = self._read_file(level)
        
        # e.g. goals = '55:17', return (55, 17)
        split_goals = lambda goals: list(map(lambda s: int(s), goals.split(':')))
        
        dfcalc = pd.concat(
            [
                df['GoalsHome'].map(split_goals).apply(pd.Series),   # (20, 2)
                df['GoalsAway'].map(split_goals).apply(pd.Series)    # (20, 2)
            ],
            axis=1
        )
        dfcalc.columns = ['HomeGoal', 'HomeConceded', 'AwayGoal', 'AwayConceded']
        dfcalc.index = df['Team'] if level == 0 else df['Name']
        dfcalc.insert(0, 'Rank', list(range(1, dfcalc.shape[0]+1)))
        
        # League average...
        lhg = dfcalc['HomeGoal'].sum() / 380      # ...Home Goals
        lhc = dfcalc['HomeConceded'].sum() / 380  # ...Home Conceded
        lag = dfcalc['AwayGoal'].sum() / 380      # ...Away Goals
        lac = dfcalc['AwayConceded'].sum() / 380  # ...Away Conceded
        
        # Teams average...
        tms_hg = dfcalc['HomeGoal'] / 19        # ...Home Goals
        tms_hc = dfcalc['HomeConceded'] / 19    # ...Home Conceded
        tms_ag = dfcalc['AwayGoal'] / 19        # ...Away Goals
        tms_ac = dfcalc['AwayConceded'] / 19    # ...Away Conceded
        
        # Attack Strength at Home/Away
        dfcalc['ASH'] = tms_hg / lhg
        dfcalc['ASA'] = tms_ag / lag
        
        # Defence Strength at Home/Away
        dfcalc['DSH'] = tms_hc / lhc
        dfcalc['DSA'] = tms_ac / lac
        
        return dfcalc
    
    def compute_result(self):
        """
        The result is based on the previous season.
        ASH : Attack Strength at Home
        ASA : Attack Strength Away
        DSH : Defence Strength at Home
        DSA : Defence Strength Away
        isFromCL : True if this team was in Championship League in the previous season
        """
        # strength are calculated based on last season
        df_pl_strength = self.calc_strength(level=0)
        df_cl_strength = self.calc_strength(level=1)
        
        # table of this season
        df = pd.read_csv(f'table/{self.season}.csv')
        
        df_merge_pl = df[['Rank', 'Team', 'Points']].merge(
            df_pl_strength[['ASH', 'ASA', 'DSH', 'DSA']], 
            left_on='Team', right_index=True, 
        )
        df_merge_pl['isFromCL'] = False

        df_merge_cl = df[['Rank', 'Team', 'Points']].merge(
            df_cl_strength[['ASH', 'ASA', 'DSH', 'DSA']], 
            left_on='Team', right_index=True, 
        )
        df_merge_cl['isFromCL'] = True

        # final result
        df_final = pd.concat((df_merge_pl, df_merge_cl), axis=0).sort_values(by='Rank')
        
        return df_final
    
    
def append_strength_to_df_team(df_team, df_strength, team_name):
    """
    append these columns from to df_team:
    SelfAS, SelfDS, SelfFromCL, RivalAS, RivalDS, RivalFromCL
    """
    # deal with the attack/defense strength of the rival team
    # merge the ASH/ASA/DSH/DSA of the rival team
    df_tmp_rvl = df_team[['Round', 'isHome', 'Rival']].merge(
        df_strength[['Team', 'ASH', 'ASA', 'DSH', 'DSA', 'isFromCL']],
        left_on='Rival', right_on='Team', how='left'
    ).drop('Team', axis=1)

    # decide to keep ASH/DSH or ASA/DSA according to the (rival team) home or away match
    df_tmp_rvl = pd.concat(
        [
            df_tmp_rvl[df_tmp_rvl.isHome].drop(['ASH', 'DSH'], axis=1).rename(
                {'ASA': 'RivalAS', 'DSA': 'RivalDS', 'isFromCL': 'RivalFromCL'}, axis=1
            ),
            df_tmp_rvl[~df_tmp_rvl.isHome].drop(['ASA', 'DSA'], axis=1).rename(
                {'ASH': 'RivalAS', 'DSH': 'RivalDS', 'isFromCL': 'RivalFromCL'}, axis=1
            )
        ]
    ).sort_values(by='Round')


    # merge the AS/DS according to the home or away match
    df_tmp_self = df_team[['Round', 'isHome']].copy()
    df_tmp_self['Self'] = team_name

    df_tmp_self = df_tmp_self.merge(
        df_strength[['Team', 'ASH', 'ASA', 'DSH', 'DSA', 'isFromCL']],
        left_on='Self', right_on='Team', how='left'
    ).drop(['Team', 'Self'], axis=1)

    # decide to keep ASH/DSH or ASA/DSA according to the home or away match
    df_tmp_self = pd.concat(
        [
            df_tmp_self[df_tmp_self.isHome].drop(['ASA', 'DSA'], axis=1).rename(
                {'ASH': 'SelfAS', 'DSH': 'SelfDS', 'isFromCL': 'SelfFromCL'}, axis=1
            ),
            df_tmp_self[~df_tmp_self.isHome].drop(['ASH', 'DSH'], axis=1).rename(
                {'ASA': 'SelfAS', 'DSA': 'SelfDS', 'isFromCL': 'SelfFromCL'}, axis=1
            )
        ]
    ).sort_values(by='Round')


    df_tmp_self = df_tmp_self[['SelfAS', 'SelfDS', 'SelfFromCL']]
    df_tmp_rvl = df_tmp_rvl[['RivalAS', 'RivalDS', 'RivalFromCL']]

    df_team = pd.concat(
        [
            df_team,
            df_tmp_self,
            df_tmp_rvl
        ],
        axis=1
    )
    return df_team
    
    
if __name__ == '__main__':
    print('Test computing the attck/defense strength at the 1819 season')
    print('------------------------------------------------------------')
    print(Strength('1819').compute_result())
    print()
    
    print('Test merging strength dataframe to the original team data')
    print('---------------------------------------------------------')
    season = '1920'
    team = 'Man City'
    test_df = append_strength_to_df_team(
        pd.read_csv(f'team_data/{season}/{team}.csv'),
        Strength(season).compute_result(),
        team
    )
    print(test_df.iloc[10,:])
    print()
    
    print('Update team data')
    print('----------------')
    for iseason in range(11):
        season = str(1011 + iseason * 101)
        print(f'[{season}] --- ', end='  ')

        df_strength = Strength(season).compute_result()
        all_teams = df_strength['Team']

        for team in all_teams:
            print(team, end=' / ')
            df_team = pd.read_csv(f'team_data/{season}/{team}.csv')
            df_team = append_strength_to_df_team(df_team, df_strength, team)
            df_team.to_csv(f'team_data/{season}/{team}.csv', index=False)

        print()


# In[6]:


for iseason in range(11):
    season = str(1011 + iseason * 101)
    print(f'[{season}] --- ', end='  ')
    
    allteams = [re.findall('([A-Za-z ]+).csv$', t)[0] for t in glob.glob(f'team_data/{season}/*.csv')]
    
    # read df of all teams
    df_allteams_pts = {}
    for team in allteams:
        df_allteams_pts[team] = pd.read_csv(f'team_data/{season}/{team}.csv')['bCumPoints']

    df_allteams_pts = pd.DataFrame(df_allteams_pts).T
    
    # mean / std cumulative points
    mean_pts = df_allteams_pts.mean()
    std_pts = df_allteams_pts.std()

    # result: standardized cumulative points
    # result.index : all teams, result.columns : round (start from 0)
    result = (df_allteams_pts - mean_pts) / std_pts
    
    # insert standardized cumulative points information into team_data
    for team in allteams:
        print(team, end=' / ')
        std_cumpts = result.loc[team,:]
        df_team = pd.read_csv(f'team_data/{season}/{team}.csv')
        df_team['bStdCumPoints'] = std_cumpts
        df_team.to_csv(f'team_data/{season}/{team}.csv', index=False)
    print()


# In[7]:


def merge_rival_info_to_df_team(team, season):
    df_team = pd.read_csv(f'team_data/{season}/{team}.csv')
    rivals = df_team['Rival'].unique()

    df_rivals = []

    for rival in rivals:
        df_rival = pd.read_csv(f'team_data/{season}/{rival}.csv')

        target_cols = df_rival.columns[df_rival.columns.str.startswith('b')]
        target_cols = target_cols.insert(0, 'Date')

        df_rivals.append(
            df_rival.loc[df_rival['Rival'] == team, target_cols]
        )

    df_rivals = pd.concat(df_rivals).sort_values(by='Date')
    df_rivals.columns = df_rivals.columns.str.replace('b', 'bRival')

    df_team = df_team.merge(df_rivals, on='Date')
    return df_team


if __name__ == '__main__':
    for iseason in range(11):
        season = str(1011 + iseason * 101)
        print(f'[{season}] --- ', end='  ')

        allteams = list(
            map(
                lambda s: s.replace('.csv', ''), 
                filter(lambda s: s.endswith('csv'), os.listdir(f'team_data/{season}/'))
            )
        )

        df_team_dict = {}    # Dict[str_team_name, df_team]

        for team in allteams:
            print(team, end=' / ')        
            df_team = merge_rival_info_to_df_team(team, season)
            df_team_dict[team] = df_team
        print()

        for team, df_team in df_team_dict.items():
            df_team.to_csv(f'team_data/{season}/{team}.csv', index=False)

